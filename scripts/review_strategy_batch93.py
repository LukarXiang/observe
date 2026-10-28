"""Audit frozen FFScore operands and literal callbacks; never admit synthetic trades."""
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
import datetime as dt
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data.financial_schema import parse_number
from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch93/20261010-ffscore-threefactor-operands')
READINESS = Path('data/staging/strategies-batch93/20261010-ffscore-threefactor-readiness')
SNAPSHOT = '20261006-145049-e06c'
SNAPSHOT_SHA = '9449dc18911fe5f27fe65f55f5849602bec0cba998335b74a6d2b1377e19af60'
SOURCES = (
    '2023年度精选策略/18.【复现】FFScore财务模型.txt',
    '2024年度精选策略2/67.三因子策略年化21%简单多因子策略框架.txt',
)
SOURCE_SHA = (
    '67873dc4ab259eb22bb79fcf98d506eb7dd285b1c54f89afa6ddefb2cf89f01b',
    '53a46f09a3a8a57415a507665e49b7a5c2fbf626489cabd38e8c4fb50939860f',
)
PARENT = Path('docs/handoff/2026-10-10-batch92-final-review.json')
PARENT_SHA = '228e098d52f9bd1c15b21c04627be4043dd51e70444defe15155accc37cdd420'
IMPORT = Path('data/raw/external_financials/c863b10a0ed2c27df694.json')
IMPORT_SHA = '95f10ec290593822f7fa23332836891d0f00299b0ad91fc5a7a647fdc9361fdd'
FILES = ('scripts/review_strategy_batch93.py', 'tests/unit/test_batch93_stage.py')
FINANCIAL_TABLES = ('financial_annual', 'financial_quarterly')
# These are dictionary candidates, not a declared mapping to JoinQuant factors.
AMOUNTS = {
    'net_operate_cash_flow': '经营活动产生的现金流量净额',
    'total_assets': '资产总计',
    'operating_revenue': '营业收入',
    'total_current_assets': '流动资产合计',
    'total_current_liability': '流动负债合计',
    'total_non_current_assets': '非流动资产合计',
    'total_non_current_liability': '非流动负债合计',
    'paidin_capital': '实收资本(或股本)',
    'total_profit': '利润总额',
    'financial_expense': '财务费用',
}
RATIO_CANDIDATES = {
    'roa': ('总资产净利润率(ROA)A', '总资产净利润率(ROA)B',
            '总资产净利润率(ROA)C', '总资产净利润率(ROA)TTM'),
    'roe': ('净资产收益率（ROE）A', '净资产收益率（ROE）B',
            '净资产收益率（ROE）C', '净资产收益率（ROE）TTM'),
    'gross_profit_margin': ('营业毛利率', '营业毛利率TTM'),
}
SCORE_FIELDS = ['DELTA_ROE', 'ROA2', 'DELTA_MARGIN', 'DELTA_CATURN',
                'DELTA_ROA2', 'DLTA_LIQUID']
METRICS = ['ROE', 'DELTA_ROE', 'ROA1', 'DELTA_ROA1', 'ROA2', 'DELTA_ROA2',
           'ACCRUAL', 'DELTA_LEVER', 'DLTA_LIQUID', 'DELTA_MARGIN', 'EQ_OFFSER',
           'DELTA_CATURN', 'DELTA_TURN', 'PB_RATIO']


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, (dt.date, pd.Timestamp)): return value.isoformat()
    if value is pd.NA: return None
    return value


def encoded(value):
    return (json.dumps(clean(value), ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False)+'\n').encode()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle: handle.write(encoded(value))


@contextmanager
def no_network():
    def forbidden(*args, **kwargs): raise AssertionError('Batch93 offline network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), \
            patch.object(socket, 'create_connection', forbidden), \
            patch.object(requests.sessions.Session, 'request', forbidden):
        yield


def source_tree(number, directory=None):
    path = Path('repo/量化策略源代码')/SOURCES[number]
    if directory is not None: path = Path(directory)/f'{number}.source'
    require(file_sha(path) == SOURCE_SHA[number], 'Batch93 source bytes changed')
    return ast.parse(read_source(path)[0])


def namespace(number, directory=None):
    # Skip platform imports/profile and all other top-level execution. The empty
    # base supplies no data or platform behavior; calc receives explicit fixtures.
    class Factor: pass
    values = dict(pd=pd, np=np, dt=dt, List=List, Tuple=Tuple, Dict=Dict,
                  Union=Union, Callable=Callable, Factor=Factor, g=SimpleNamespace(),
                  log=SimpleNamespace(info=lambda *a: None, set_level=lambda *a: None))
    nodes = [n for n in source_tree(number, directory).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 '<SHA-pinned-Batch93-original-definitions>', 'exec'), values)
    return values


def parent_binding():
    require(file_sha(PARENT) == PARENT_SHA, 'Accepted92 receipt changed')
    doc = read(PARENT)
    require(doc['status'] == 'ok' and doc['formal_source_research_acceptance'] and
            not doc['formal_trading_acceptance'], 'Accepted92 scope changed')
    for prefix in ('receipt', 'checked_state', 'final_binding'):
        require(file_sha(doc[prefix+'_file']) == doc[prefix+'_sha256'], 'Accepted92 binding changed')
    checked = read(doc['checked_state_file'])
    for path, sha in doc['reviewed_implementation_sha256'].items():
        require(file_sha(path) == sha, 'Accepted92 frozen implementation changed')
    for name, sha in checked['evidence_sha256'].items():
        require(file_sha(Path(doc['accepted_directory'])/name) == sha, 'Accepted92 evidence changed')
    return {'file': str(PARENT), 'sha256': PARENT_SHA,
            'checked_evidence_files': len(checked['evidence_sha256']),
            'checked_implementation_files': len(doc['reviewed_implementation_sha256'])}


def operand_inventory(fields, projection, availability, state):
    coverage = []
    for table in FINANCIAL_TABLES:
        data = projection[projection.archive_table.eq(table)]
        for key, labels in {**{k: (v,) for k, v in AMOUNTS.items()}, **RATIO_CANDIDATES}.items():
            for label in labels:
                number, malformed = parse_number(data[label])
                dictionary = fields[fields.table.eq(table) & fields.column.eq(label)]
                require(len(dictionary) == 1, f'Ambiguous dictionary operand: {table}/{label}')
                coverage.append({'table': table, 'platform_dependency_candidate': key,
                    'column': label, 'rows': len(data), 'finite_values': int(number.notna().sum()),
                    'malformed_nonmissing': int(malformed.sum()), 'zeros': int(number.eq(0).sum()),
                    'positive_values': int(number.gt(0).sum()),
                    'first_report_period': data.loc[number.notna(), 'report_period'].min(),
                    'last_report_period': data.loc[number.notna(), 'report_period'].max(),
                    'dictionary': dictionary.iloc[0].to_dict(), 'platform_equivalence': 'not_proven',
                    'strict_usable_values': 0})
    ns = namespace(0)
    dependencies = ns['F13_Score'].dependencies
    require(len(availability) and not availability.strict_usable.any(), 'Strict coverage changed; replan')
    return {'snapshot': SNAPSHOT, 'projection_rows': len(projection), 'field_rows': len(fields),
        'table_rows': projection.archive_table.value_counts().to_dict(),
        'matched_instrument_rows': int(projection.instrument.notna().sum()),
        'availability_rows': len(availability), 'strict_usable_rows': 0,
        'version_evidence': sorted(availability.version_evidence.unique()),
        'raw_dependencies': dependencies, 'unique_dependencies': list(dict.fromkeys(dependencies)),
        'duplicated_dependencies': {k: n for k, n in Counter(dependencies).items() if n > 1},
        'coverage': coverage, 'registered_tables': sorted(state['tables']),
        'financial_history_registered_fields': 'CORE only; most FFScore raw operands are not admitted',
        'platform_style_factor_table_registered': any(t in state['tables'] for t in
            ('jqfactor', 'style_factors', 'factor_values')),
        'limits': ['All final values remain revision-unknown; report period is not availability',
            'Ratio A/B/C/TTM are distinct, unitless vendor definitions; original ROA assumes percent',
            'paidin_capital candidate is yuan, not a proven historical number of shares',
            '_1/_4/_5 need platform dependency period/version definitions, not row shifts',
            'Historical whole pool, security-info date semantics, ST and SW_l1 identities unproved',
            'Daily PB candidate does not establish full-pool platform PB and neutralization equality'],
        'original_backtests': 0, 'formal_acceptance': False}


def synthetic_operands():
    ns = namespace(0); cols = ['A', 'B', 'C']
    values = {key: [10., 10., 10.] for key in set(ns['F13_Score'].dependencies)}
    values.update(total_profit=[100., -100., 0.], financial_expense=[20., 20., 20.],
        total_profit_4=[50., -50., 0.], financial_expense_4=[10., 10., 10.],
        total_current_assets=[100., 100., 100.], total_current_assets_1=[100., 100., 100.],
        total_current_assets_4=[100., 100., 100.], total_current_assets_5=[100., 100., 100.],
        operating_revenue=[200., 100., 100.], operating_revenue_4=[100., 100., 100.],
        roe=[20., 10., 10.], roe_4=[10., 10., 10.],
        gross_profit_margin=[20., 10., 10.], gross_profit_margin_4=[10., 10., 10.],
        total_current_liability=[25., 50., 50.], total_current_liability_4=[50., 50., 50.])
    return {k: pd.DataFrame([v], columns=cols) for k, v in values.items()}


def independent_one_row(data):
    result = []
    for col in next(iter(data.values())).columns:
        d = {k: np.float64(v[col].iloc[0]) for k, v in data.items()}
        require(all(len(v) == 1 for v in data.values()), 'Independent check needs one dependency row')
        with np.errstate(divide='ignore', invalid='ignore'):
            assets = d['total_current_assets']+d['total_current_assets_1']
            old_assets = d['total_current_assets_4']+d['total_current_assets_5']
            profit = d['total_profit']+d['financial_expense']/assets
            old_profit = d['total_profit_4']+d['financial_expense_4']/old_assets
            vals = [d['roe'], d['roe']/d['roe_4']-1, d['roa'], d['roa']/d['roa_4']-1,
                profit, profit/old_profit-1,
                d['net_operate_cash_flow']/d['total_assets']-d['roa']*.01,
                -(d['total_non_current_liability']/d['total_non_current_assets']/
                  (d['total_non_current_liability_4']/d['total_non_current_assets_4'])-1),
                d['total_current_assets']/d['total_current_liability']/
                (d['total_current_assets_4']/d['total_current_liability_4'])-1,
                d['gross_profit_margin']/d['gross_profit_margin_4']-1,
                d['paidin_capital']/d['paidin_capital_4']-1,
                (d['operating_revenue']/assets)/(d['operating_revenue_4']/old_assets)-1,
                (d['operating_revenue']/(d['total_assets']+d['total_assets_1']))/
                (d['operating_revenue_4']/(d['total_assets_4']+d['total_assets_5']))-1,
                d['pb_ratio']]
        result.append([v if np.isfinite(v) else np.nan for v in vals])
    return pd.DataFrame(result, index=next(iter(data.values())).columns, columns=METRICS)


def evaluate_fixture(data, directory=None):
    ns = namespace(0, directory); ns['set_params'](); f = ns['F13_Score']()
    score = f.calc(data); independent = independent_one_row(data)
    pd.testing.assert_frame_equal(f.basic, independent, check_exact=True)
    expected = independent[SCORE_FIELDS].gt(0).sum(axis=1)
    require(score.astype(int).equals(expected), 'Independent six-field scores differ')
    return {'basic': f.basic.to_dict('index'), 'scores': score.to_dict(),
            'independent_metric_cells': int(f.basic.size), 'differences': 0}


def factor_diagnostics(directory=None):
    base = synthetic_operands(); outputs = []
    outputs.append({'case': 'one_row_three_securities', **evaluate_fixture(base, directory)})
    nan = {k: v.copy() for k, v in base.items()}
    for value in nan.values(): value.loc[0, 'C'] = np.nan
    outputs.append({'case': 'all_nan_security_scores_zero', **evaluate_fixture(nan, directory)})
    zero = {k: v.copy() for k, v in base.items()}
    for value in zero.values(): value.loc[0, 'C'] = 0.
    outputs.append({'case': 'zero_denominators_inf_to_nan', **evaluate_fixture(zero, directory)})
    scaled = {k: v.copy() for k, v in base.items()}
    for k, v in scaled.items():
        if k.removesuffix('_1').removesuffix('_4').removesuffix('_5') in AMOUNTS: v *= 1000
    outputs.append({'case': 'monetary_scale_1000_ROA2_not_invariant', **evaluate_fixture(scaled, directory)})
    ns = namespace(0, directory); ns['set_params'](); factor = ns['F13_Score']()
    two = {k: pd.concat([v, v], ignore_index=True) for k, v in base.items()}
    try: factor.calc(two)
    except ValueError as exc:
        outputs.append({'case': 'two_dependency_rows_rejected', 'exception': type(exc).__name__,
                        'message': str(exc), 'reason': '14 concat operands produce28 columns; original assigns14'})
    else: raise AssertionError('Two-row compatibility defect changed')
    return outputs


def filter_diagnostics(directory=None):
    ns = namespace(0, directory); outputs = []
    for label, ends in (
            ('string_end_date', ['2200-01-01', '2020-01-01']),
            ('datetime64_end_date', pd.to_datetime(['2200-01-01', '2020-01-01'])),
            ('date_object_end_date', [dt.date(2200, 1, 1), dt.date(2020, 1, 1)])):
        ns['get_all_securities'] = lambda **kw: pd.DataFrame({'end_date': ends}, index=['live', 'ended'])
        obj = ns['Filter_Stocks']('A', '2024-06-03')
        outputs.append({'case': label, 'selected': obj.securities,
                        'platform_return_dtype_proven': False})
    obj = ns['Filter_Stocks'].__new__(ns['Filter_Stocks'])
    obj.watch_date = dt.date(2024, 6, 3)
    columns = ['paused20', 'paused21', 'paused22', 'nan_paused']
    paused = pd.DataFrame({columns[0]: [1]*20+[0]*2, columns[1]: [1]*21+[0],
                          columns[2]: [1]*22, columns[3]: [np.nan]*22})
    rows = paused.assign(time=pd.date_range('2024-05-01', periods=22)).melt(id_vars='time', var_name='code', value_name='paused')
    ns['get_price'] = lambda *a, **kw: rows
    obj.securities = columns; obj.filter_paused(22, 21)
    outputs.append({'case': 'paused22_threshold21_and_missing_sum', 'selected': obj.securities,
                    'missing_is_not_real_eligible_data': True})
    try: obj.filter_paused(22, 23)
    except ValueError as exc: outputs.append({'case': 'paused_threshold23_error', 'exception': type(exc).__name__})
    else: raise AssertionError('Paused parameter guard changed')
    obj.securities = ['age180', 'age181', 'unknown']
    ns['get_security_info'] = lambda c: None if c == 'unknown' else SimpleNamespace(
        start_date=obj.watch_date-dt.timedelta(days=int(c[3:])))
    obj.filter_ipodate(180)
    outputs.append({'case': 'IPO180_strict_boundary', 'selected': obj.securities})
    obj.securities = ['ok', 'st', 'unknown']
    ns['get_extras'] = lambda *a, **kw: pd.DataFrame([[False, True, np.nan]], columns=obj.securities)
    obj.filter_st(); outputs.append({'case': 'ST_missing_not_false', 'selected': obj.securities})
    obj.securities = ['bank', 'nonbank', 'industry_missing', 'other']
    ns['get_industry'] = lambda *a: {
        'bank': {'sw_l1': {'industry_name': '金融服务I'}},
        'nonbank': {'sw_l1': {'industry_name': '非银金融I'}},
        'industry_missing': {}, 'other': {'sw_l1': {'industry_name': '制造I'}}}
    obj.filter_industry(['金融服务I', '非银金融I'])
    outputs.append({'case': 'industry_exclusion_and_missing', 'selected': obj.securities})
    return outputs


def pb_diagnostics(directory=None):
    ns = namespace(0, directory); calls = []
    def transform(name):
        def fn(value, **kw): calls.append({'function': name, 'keywords': kw}); return value
        return fn
    for name in ('winsorize_med', 'standardlize', 'neutralize'): ns[name] = transform(name)
    original = pd.DataFrame({'code': ['A', 'B', 'C'], 'pb_ratio': [1., 2., 3.]})
    output = ns['prepare_pb_ratio'](original, dt.date(2024, 6, 3))
    return [{'case': 'PB_call_order_identity_spies_only', 'calls': calls,
             'values': output.to_dict(), 'real_platform_transformation': False}]


def ff_trade_diagnostics(directory=None):
    outputs = []
    for label, scores, pb in (
            ('empty_target_after_sell', [4, 4, 4], [1., 2., 3.]),
            ('intersection_one_target', [6, 5, 4], [1., 2., 3.])):
        ns = namespace(0, directory); ns['set_params'](); calls = []
        class Pool:
            securities = ['A', 'B', 'C']
            def __init__(self, *a): pass
            def filter_paused(self, *a): calls.append(['paused', *a])
            def filter_st(self): calls.append(['st'])
            def filter_ipodate(self, *a): calls.append(['IPO', *a])
            def filter_industry(self, *a): calls.append(['industry', *a])
        ns['Filter_Stocks'] = Pool
        ns['calc_factors'] = lambda *a, **kw: {'F13_Score': pd.DataFrame([scores], columns=Pool.securities)}
        ns['get_valuation'] = lambda *a, **kw: pd.DataFrame({'code': Pool.securities, 'pb_ratio': pb})
        for name in ('winsorize_med', 'standardlize', 'neutralize'): ns[name] = lambda x, **kw: x
        orders = []
        ns['order_target'] = lambda *a: orders.append(['target', *a])
        ns['order_target_value'] = lambda *a: orders.append(['target_value', *a])
        ns['record'] = lambda **kw: calls.append(['record', kw])
        context = SimpleNamespace(previous_date=dt.date(2024, 6, 3),
            portfolio=SimpleNamespace(long_positions={'old': object()}, total_value=100000.))
        error = None
        try: ns['trade'](context)
        except ZeroDivisionError as exc: error = type(exc).__name__
        outputs.append({'case': label, 'filters': calls, 'order_intents': orders, 'exception': error,
                        'actual_orders': 0, 'identity_PB_spies_not_factor_engine': True})
    return outputs


def commission_diagnostics(directory=None):
    ns = namespace(0, directory); output = []; events = []
    ns['FixedSlippage'] = lambda x: x
    ns['PerTrade'] = lambda **kw: kw
    ns['set_slippage'] = lambda x: events.append(['slippage', x])
    ns['set_commission'] = lambda x: events.append(['commission', x])
    for date in ('2009-01-01T00:00:00', '2009-01-01T09:00:00', '2011-01-01T00:00:00',
                 '2011-01-01T09:00:00', '2013-01-01T00:00:00', '2013-01-01T09:00:00'):
        events.clear(); ns['before_trading_start'](SimpleNamespace(current_dt=dt.datetime.fromisoformat(date)))
        output.append({'case': date, 'direct_callback_events': list(events), 'platform_auto_schedule_proven': False})
    return output


def threefactor_selection_diagnostics(directory=None):
    outputs = []
    for case in ('month_change', 'same_month', 'same_month_other_year', 'permuted_res_columns'):
        ns = namespace(1, directory); calls = []
        stocks = [f'S{i:02d}' for i in range(55)]
        ns['get_all_securities'] = lambda **kw: pd.DataFrame(index=stocks)
        ns['get_extras'] = lambda *a, **kw: pd.DataFrame([[False]*54+[True]], columns=stocks)
        def factors(selected, names, **kw):
            calls.append({'function': 'get_factor_values', 'stocks': selected, 'factors': names, 'keywords': kw})
            data = {name: pd.DataFrame([np.arange(len(selected), dtype=float)], columns=selected) for name in names}
            data['growth'].iloc[0] = np.arange(len(selected))[::-1]
            data['momentum'].iloc[0] = 99999.
            data['VROC12'].iloc[0, 0] = np.nan
            if case == 'permuted_res_columns':
                data['residual_volatility'] *= 100
                data['residual_volatility'] = data['residual_volatility'].iloc[:, ::-1]
            return data
        ns['get_factor_values'] = factors
        def std(value, **kw): calls.append({'function': 'standardlize', 'keywords': kw}); return np.asarray(value)+10
        def wins(value, **kw): calls.append({'function': 'winsorize', 'keywords': kw}); return np.asarray(value)*2
        ns['standardlize'] = std; ns['winsorize'] = wins
        ns['g'].securities = ['held_prior']
        current = dt.datetime(2024, 6, 3, 9)
        previous = dt.date(2024, 5, 31) if case in ('month_change', 'permuted_res_columns') else (
            dt.date(2023, 6, 3) if case == 'same_month_other_year' else dt.date(2024, 6, 2))
        ns['before_market_open'](SimpleNamespace(current_dt=current, previous_date=previous))
        outputs.append({'case': case, 'selected': ns['g'].securities, 'calls': calls,
            'transforms': 'controlled +10 then*2 spies; not real winsorization/style factors',
            'column_alignment': 'original uses positional values under growth.columns'})
    return outputs


def threefactor_order_diagnostics(directory=None):
    outputs = []
    for case in ('zero_default_positions', 'plain_dict_positions', 'same_month_no_orders',
                 'sell_updates_cash_before_read'):
        ns = namespace(1, directory); ns['g'].securities = ['A', 'B']; orders = []
        zero = lambda: SimpleNamespace(closeable_amount=0)
        positions = {'old': SimpleNamespace(closeable_amount=100), 'B': SimpleNamespace(closeable_amount=100)}
        if case != 'plain_dict_positions': positions = defaultdict(zero, positions)
        portfolio = SimpleNamespace(positions=positions, available_cash=10000.)
        def sell(code, amount):
            orders.append(['target', code, amount])
            if case == 'sell_updates_cash_before_read': portfolio.available_cash += 5000.
        ns['order_target'] = sell
        ns['order_value'] = lambda code, value: orders.append(['value', code, value])
        previous = dt.date(2024, 6, 2) if case == 'same_month_no_orders' else dt.date(2024, 5, 31)
        context = SimpleNamespace(current_dt=dt.datetime(2024, 6, 3), previous_date=previous, portfolio=portfolio)
        error = None
        try: ns['market_open'](context)
        except KeyError as exc: error = f'{type(exc).__name__}:{exc.args[0]}'
        outputs.append({'case': case, 'order_intents': orders, 'exception': error,
            'actual_orders': 0, 'platform_missing_position_and_immediate_fill_semantics_proven': False})
    return outputs


def diagnostics(directory=None):
    groups = {'FFScore_arithmetic': factor_diagnostics(directory),
        'FFScore_filters': filter_diagnostics(directory), 'PB_pipeline': pb_diagnostics(directory),
        'FFScore_order_intents': ff_trade_diagnostics(directory),
        'commission_direct_callback': commission_diagnostics(directory),
        'threefactor_selection': threefactor_selection_diagnostics(directory),
        'threefactor_order_intents': threefactor_order_diagnostics(directory)}
    return {'groups': groups, 'cases': sum(map(len, groups.values())),
        'independent_metric_cells': sum(r.get('independent_metric_cells', 0) for r in groups['FFScore_arithmetic']),
        'new_trading_backtests': 0, 'real_platform_factors': 0, 'formal_acceptance': False,
        'scope': 'literal original definitions with explicit synthetic operands/callback spies; no real execution'}


def stage(root, directory):
    require(not directory.exists(), 'Batch93 stage already exists; never overwrite')
    bound = parent_binding()
    require(file_sha(READINESS/'source-readiness.json') ==
            'b5797e5927a630710e3790710f90de5da66beab2691ad596eb1b1f9158d931ff', 'Readiness changed')
    require(file_sha(Path(root)/'snapshots'/f'{SNAPSHOT}.json') == SNAPSHOT_SHA, 'Frozen snapshot changed')
    require(file_sha(IMPORT) == IMPORT_SHA, 'Financial import byte manifest changed')
    for n in range(2): source_tree(n); source_tree(n, READINESS)
    store = Store(root); state = store.state(SNAPSHOT)
    require(store.published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    implementation = {name: file_sha(name) for name in FILES}
    checkpoint(root, directory, 'Batch93 operand/callback audit started; no new original signals or trades')
    for n in range(2): shutil.copyfile(READINESS/f'{n}.source', directory/f'{n}.source')
    for name in FILES:
        copied = directory/'implementation'/name; copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, copied)
        require(file_sha(copied) == implementation[name], 'Candidate changed during copy')
    byte_manifest = read(IMPORT)['partition_bytes']; partitions = []
    tables = (*FINANCIAL_TABLES, 'financial_fields', 'financial_availability')
    for table in tables:
        for entry in state['tables'][table].values():
            path = Path(root)/entry['file']; before = path.stat(); actual = file_sha(path)
            require(actual == byte_manifest[entry['file']], 'Frozen financial partition bytes changed')
            after = path.stat()
            require(before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
                    'Financial input changed during hashing')
            partitions.append({'table': table, 'file': str(path), 'bytes_sha256': actual,
                'snapshot_logical_sha': entry['sha'], 'rows': entry['rows'],
                'size': after.st_size, 'mtime_ns': after.st_mtime_ns})
    fields = store.load_state(state, 'financial_fields')
    require(fingerprint(fields) == next(iter(state['tables']['financial_fields'].values()))['sha'],
            'Dictionary logical fingerprint changed')
    raw_columns = [*AMOUNTS.values(), *(v for labels in RATIO_CANDIDATES.values() for v in labels)]
    columns = ['instrument', 'source_sha256', 'source_row', 'report_period', 'version_evidence', *raw_columns]
    pieces = []
    for table in FINANCIAL_TABLES:
        frame = store.load_state(state, table, columns=columns); frame.insert(0, 'archive_table', table)
        require(len(frame) == sum(e['rows'] for e in state['tables'][table].values()), 'Projection row count changed')
        pieces.append(frame)
    projection = pd.concat(pieces, ignore_index=True)
    availability = store.load_state(state, 'financial_availability', columns=['strict_usable', 'version_evidence'])
    fields.to_parquet(directory/'financial-fields.parquet', index=False)
    projection.to_parquet(directory/'financial-operands.parquet', index=False)
    save_new(directory/'input-binding.json', {'snapshot': SNAPSHOT, 'snapshot_sha256': SNAPSHOT_SHA,
        'parent': bound, 'financial_import': str(IMPORT), 'financial_import_sha256': IMPORT_SHA,
        'source_sha256': SOURCE_SHA, 'partitions': partitions, 'implementation_sha256': implementation,
        'projection_columns': columns, 'projection_bytes_sha256': file_sha(directory/'financial-operands.parquet'),
        'dictionary_bytes_sha256': file_sha(directory/'financial-fields.parquet'), 'not_a_backtest': True})
    failure = Path('data/staging/strategies-batch93/20261010-development-first-fixtures')
    save_new(directory/'development-failure-binding.json', {str(p): file_sha(p)
        for p in sorted(failure.iterdir()) if p.is_file()})
    save_new(directory/'operand-inventory.json', operand_inventory(fields, projection, availability, state))
    save_new(directory/'original-definition-review.json', {'sources': [
        {'path': SOURCES[n], 'sha256': SOURCE_SHA[n], 'lines': len(read_source(directory/f'{n}.source')[0].splitlines()),
         'definition_ast': [ast.dump(node, include_attributes=False) for node in source_tree(n, directory).body
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef))]} for n in range(2)],
        'selected_fields': SCORE_FIELDS, 'top_level_executed': False,
        'formal_source_acceptance': False, 'new_trading_backtests': 0})
    return {'status': 'operands_frozen', 'rows': len(projection), 'fresh_byte_verified_partitions': len(partitions)}


def verify_binding(directory):
    binding = read(directory/'input-binding.json'); parent_binding()
    for n in range(2): source_tree(n, directory)
    for path, sha in binding['implementation_sha256'].items():
        require(file_sha(path) == file_sha(directory/'implementation'/path) == sha, 'Frozen candidate changed')
    require(file_sha(directory/'financial-operands.parquet') == binding['projection_bytes_sha256'], 'Projection changed')
    require(file_sha(directory/'financial-fields.parquet') == binding['dictionary_bytes_sha256'], 'Dictionary changed')
    for p in binding['partitions']:
        stat = Path(p['file']).stat()
        require(stat.st_size == p['size'] and stat.st_mtime_ns == p['mtime_ns'], 'Operand partition changed')
    return binding


def diagnose(directory):
    verify_binding(directory)
    require(not (directory/'diagnostics.json').exists(), 'Diagnostics already frozen')
    with no_network(): result = diagnostics(directory)
    save_new(directory/'diagnostics.json', result)
    return {'status': 'diagnostics_frozen', 'cases': result['cases'],
            'independent_metric_cells': result['independent_metric_cells']}


def offline(directory):
    verify_binding(directory)
    require(not (directory/'offline-verification.json').exists(), 'Offline receipt already exists')
    with no_network(): result = diagnostics(directory)
    require(encoded(result) == (directory/'diagnostics.json').read_bytes(), 'Offline callback replay differs')
    save_new(directory/'offline-verification.json', {'status': 'ok', 'cases': result['cases'],
        'independent_metric_cells': result['independent_metric_cells'], 'differences': 0,
        'network': 'socket.connect/create_connection and requests.Session.request forbidden',
        'diagnostic_sha256': file_sha(directory/'diagnostics.json'),
        'new_trading_backtests': 0, 'formal_acceptance': False})
    return {'status': 'ok', 'differences': 0}


def supplements(directory):
    """Reuse bounded failure evidence; inspect actual APIs and reference definitions."""
    import baostock as bs
    verify_binding(directory)
    require(not (directory/'supplement-decision.json').exists(), 'Supplement decision frozen')
    previous_path = Path('data/staging/strategies-batch31/20261006-capm-weekly-rsrs-finance/probe-results.json')
    previous_receipt = read('docs/handoff/2026-10-06-batch31-verification.json')
    require(file_sha(previous_path) == previous_receipt['checks']['evidence_sha256']['probe-results.json'],
            'Accepted financial supplement changed')
    prior = read(previous_path)
    references = []
    for name in ('repo/skfolio/src/skfolio/descriptor/_profitability/_return_on_assets.py',
                 'repo/skfolio/src/skfolio/descriptor/_profitability/_return_on_equity.py',
                 'repo/skfolio/src/skfolio/descriptor/_volatility/_ew_residual_volatility.py'):
        copied = directory/'references'/Path(name).name; copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, copied)
        references.append({'file': name, 'copy': str(copied), 'sha256': file_sha(copied),
            'scope': 'Definition comparison only; not admitted as original JQ ratio/style factor'})
    apis = []
    for fn in (bs.query_profit_data, bs.query_balance_data, bs.query_cash_flow_data, bs.query_growth_data):
        code = inspect.getsource(fn)
        apis.append({'function': fn.__name__, 'signature': str(inspect.signature(fn)), 'source': code,
            'sha256': hashlib.sha256(code.encode()).hexdigest()})
    save_new(directory/'supplement-decision.json', {'previous_file': str(previous_path),
        'previous_sha256': file_sha(previous_path), 'previous_receipt_sha256': file_sha('docs/handoff/2026-10-06-batch31-verification.json'),
        'previous_probes': prior, 'installed_baostock_version': bs.__version__, 'apis': apis,
        'references': references, 'new_requests': 0, 'new_samples': 0, 'published': False,
        'decision': 'Reuse accepted same-host login/DNS failures; quarterly APIs do not establish historical vintages, whole-pool SW_l1 or JQ style-factor definitions',
        'provider_absence_proven': False, 'new_trading_backtests': 0})
    return {'status': 'saved', 'new_requests': 0, 'references': len(references)}


def finish(root, directory):
    require(not (directory/'stage-checkpoint.json').exists() and not (directory/'protection.json').exists(),
            'Stage finish already written; never overwrite')
    verify_binding(directory)
    offline_proof = read(directory/'offline-verification.json')
    require(offline_proof['status'] == 'ok' and
            offline_proof['diagnostic_sha256'] == file_sha(directory/'diagnostics.json'), 'Offline replay missing or changed')
    require((directory/'supplement-decision.json').is_file(), 'Supplement evidence missing')
    protection = protect(root, directory)
    result = archive(root, 'Batch93 501353 final financial operand rows and literal synthetic callbacks audited; historical versions/style factors gated; no new trading')
    save_new(directory/'stage-checkpoint.json', {'progress': result, 'protection': protection,
        'formal_acceptance': False, 'new_trading_backtests': 0})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('stage', 'diagnose', 'offline', 'supplements', 'finish'))
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    operation = globals()[args.phase]
    value = operation(args.root, args.directory) if args.phase in ('stage', 'finish') else operation(args.directory)
    print(json.dumps(clean(value), ensure_ascii=False, indent=2))
