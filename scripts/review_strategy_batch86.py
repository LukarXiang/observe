"""Audit literal volume/price kernels; historical universes and wizard inputs gate trades."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
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
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies, read_source, strategy_id
from scripts import review_strategy_batch85 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.review_strategy_batch61 import OPERAND, OPERAND_SHA, CALENDAR, CALENDAR_SHA
from scripts.review_strategy_batch83 import definitions, selected_nodes, plain
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch86/20261009-longterm-volume-technical-flow')
SNAPSHOT = previous.SNAPSHOT
SOURCES = ('2020年度精选策略/54 中长线买入卖出点选择.txt', '2020年度精选策略/95 资金流策略.txt')
SOURCE_SHA = ('8c22e7eab356b32b4df1d496cd4bb25430ccd7b355a8f855fd609823a1a58bd0',
    'a6c3dea6c7bd5c302ec2920386e68fc4b84a212c64c98019f540642baf5a8e47')
HELPERS = ('MACD_judge_jincha', 'BBands_lowerband_judge_dayu', 'MA_judge_duotou',
    'money_flow_dayu', 'n_day_chg_dayu', 'MACD_judge_sicha', 'get_sort_dataframe',
    'sell_by_amount_or_percent_or_none', 'order_style', 'judge_security_max_proportion', 'max_buy_value_or_amount')
REFERENCES = (('repo/backtrader', 'backtrader/indicators/basicops.py'),
    ('repo/akshare', 'akshare/stock/stock_fund_em.py'),
    ('repo/rqalpha', 'rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py'))
FILES = tuple(sorted(set(previous.FILES) | {'scripts/review_strategy_batch86.py', 'tests/unit/test_batch86_research.py',
    'scripts/review_strategy_batch61.py', 'scripts/review_strategy_batch60.py', 'scripts/review_strategy_batch52.py'}))
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
    'sources-frozen.json', 'source-reviews/review.json', 'dependency-inventory.json', 'supplement-decision.json',
    'dependency-inventory-corrected.json', 'inventory-row-correction.json',
    'diagnostics.json', 'component-research.json', 'component-values.parquet', 'price-input.parquet',
    'calendar-input.parquet', 'offline-verification.json', 'offline-catalog.json'}


def implementation(): return {n: file_sha(n) for n in FILES}


def binding(root, directory=None):
    receipt = Path('docs/handoff/2026-10-09-batch85-verification.json')
    review = Path('docs/handoff/2026-10-09-batch85-final-review.json')
    accepted = read(review); proof = read(previous.DIRECTORY/'final-binding-verification.json')
    checked = read(previous.DIRECTORY/'checked-state.json')
    require(read(receipt)['status'] == accepted['status'] == proof['status'] == 'ok' and
        file_sha(receipt) == accepted['receipt_sha256'] == proof['receipt_sha256'] and
        file_sha(previous.DIRECTORY/'checked-state.json') == accepted['checked_state_sha256'] == proof['checked_state_sha256'] and
        file_sha(previous.DIRECTORY/'final-binding-verification.json') == accepted['final_binding_sha256'], 'Accepted85 binding changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted85 evidence changed')
    require(file_sha('scripts/review_strategy_batch85.py') == accepted['implementation_sha256']['scripts/review_strategy_batch85.py'], 'Parent entrypoint changed')
    price_receipt = Path('docs/handoff/2026-10-07-batch47-verification.json'); old = read(price_receipt)['checks']['evidence_sha256']
    require(file_sha(OPERAND) == OPERAND_SHA == old['price-input.parquet'] and
        file_sha(CALENDAR) == CALENDAR_SHA == old['calendar-input.parquet'], 'Accepted operands changed')
    result = {'snapshot': SNAPSHOT, 'receipt_sha256': file_sha(receipt), 'review_sha256': file_sha(review),
        'price_receipt_sha256': file_sha(price_receipt), 'price_file': str(OPERAND), 'price_sha256': OPERAND_SHA,
        'calendar_file': str(CALENDAR), 'calendar_sha256': CALENDAR_SHA, 'not_a_backtest': True}
    if directory is not None:
        require(result == read(directory/'input-binding.json'), 'Input binding changed')
        for name, sha in (('price', OPERAND_SHA), ('calendar', CALENDAR_SHA)):
            require(file_sha(directory/(name+'-input.parquet')) == sha, 'Operand copy changed')
    return result


def adapted_source(text, number):
    lines = text.splitlines(); require(len(lines) > 8 and lines[8].startswith(('# ', 'from kuanke.wizard')), 'Source boundary changed')
    code = '\n'*8+'\n'.join(lines[8:])+'\n'
    if number == 0:
        # Only print syntax is adapted; imports and missing injected names stay untouched.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            from lib2to3.refactor import RefactoringTool
        code = str(RefactoringTool(['lib2to3.fixes.fix_print']).refactor_string(code, 'frozen54'))
        require(len(code.splitlines()) == len(lines), 'Print adaptation shifted source lines')
    return code


def tree_for(directory, number):
    row = read(directory/'sources-frozen.json')[number]
    require(file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return ast.parse(adapted_source(read_source(Path(row['source_copy']))[0], number))


def selected(directory, number, names, namespace):
    return selected_nodes(tree_for(directory, number), names, namespace)


def inventory(root):
    state = Store(root).state(SNAPSHOT)
    command = ['rg', '-n', '--hidden', '--no-ignore', '^def ('+'|'.join(HELPERS)+')\\(',
        'repo', '-g', '*.py', '-g', '*.txt', '-g', '!**/.git/**']
    found = subprocess.run(command, capture_output=True, text=True)
    require(found.returncode in (0, 1), 'Helper search failed')
    return {'snapshot': SNAPSHOT, 'registered_tables': sorted(state['tables']),
        'table_rows': {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()},
        'helper_search': {'command': command, 'returncode': found.returncode, 'matches': found.stdout.splitlines(),
            'matched_file_sha256': {n: file_sha(n) for n in sorted({s.split(':', 1)[0] for s in found.stdout.splitlines()})}},
        'required_helpers95': list(HELPERS), 'reference_rows': len(pd.read_parquet(OPERAND)),
        '54_active_money_flow_calls': 0, 'strict_financial_usable_rows': 0,
        'limits': ['Ten frozen reference stocks are not historical CSI300 or bank399951 members',
            'Historical ST/names/limits, full universe price-volume and platform skip-paused/adjustment-volume rules not proved',
            'Source54 lacks mean/timedelta and imports removed pandas.stats.api.ols; platform injections unknown',
            'Source54 requests money but active volume/MA kernels do not read it; do not synthesize missing money',
            'Supplier turnover decimal exists in bars_1d; its floating-share denominator/version is not proved original valuation.turnover_ratio',
            '95 helper seeds/current-bar/threshold/flow vintage and nonempty zero-weight sorting output remain unknown',
            'Original order_target quantities and incremental wizard orders/fills require unique Book and explicit compatibility review']}


def supplements():
    path = previous.DIRECTORY/'probe-results.json'; receipt = read('docs/handoff/2026-10-09-batch85-verification.json')
    require(file_sha(path) == receipt['checks']['evidence_sha256']['probe-results.json'], 'Prior supplement changed')
    rows = read(path)['results']
    require(all(r['status'] == 'failed' and not r['wire_responses'] for r in rows), 'Prior DNS evidence changed')
    return {'file': str(path), 'sha256': file_sha(path), 'results': rows, 'api_evidence_file': str(previous.DIRECTORY/'existing-apis.json'),
        'api_evidence_sha256': file_sha(previous.DIRECTORY/'existing-apis.json'),
        'new_requests': 0, 'new_rows': 0, 'published': False,
        'decision': 'Reuse verified same-host DNS failures and recent-only flow API limitation; changing symbol cannot recover original long history. No blind retries.',
        'missing': ['Original daily CSI300/bank399951 members and history of names/ST/limits',
            'Original previous-natural-day turnover_ratio query and available denominator/vintage',
            'Exact wizard technical/flow/ranking/allocation helper version and long previous-day net_pct_main'],
        'not_a_backtest': True}


def correct_inventory(root, directory):
    binding(root, directory); validate_sources(directory)
    original = read(directory/'dependency-inventory.json'); corrected = inventory(root)
    require(original['reference_rows'] == 47781 and corrected['reference_rows'] == 46781 and
        {**original, 'reference_rows': 46781} == corrected, 'Unexpected inventory correction')
    save(directory/'dependency-inventory-corrected.json', corrected)
    save(directory/'inventory-row-correction.json', {'original_file': str(directory/'dependency-inventory.json'),
        'original_sha256': file_sha(directory/'dependency-inventory.json'),
        'corrected_file': str(directory/'dependency-inventory-corrected.json'),
        'corrected_sha256': file_sha(directory/'dependency-inventory-corrected.json'),
        'price_sha256': OPERAND_SHA, 'old_count': 47781, 'actual_count': 46781,
        'explanation': 'Initial handoff prose count was incorrect; pinned accepted payload did not change', 'not_a_backtest': True})
    return {'status': 'corrected', 'actual_rows': 46781}


def validate_inventory(root, directory):
    corrected = inventory(root); correction = read(directory/'inventory-row-correction.json')
    require(corrected == read(directory/'dependency-inventory-corrected.json') and
        correction['original_file'] == str(directory/'dependency-inventory.json') and
        correction['corrected_file'] == str(directory/'dependency-inventory-corrected.json') and
        file_sha(directory/'dependency-inventory.json') == correction['original_sha256'] and
        file_sha(directory/'dependency-inventory-corrected.json') == correction['corrected_sha256'] and
        correction['price_sha256'] == OPERAND_SHA and correction['old_count'] == 47781 and correction['actual_count'] == 46781 and
        {**read(directory/'dependency-inventory.json'), 'reference_rows': 46781} == corrected, 'Inventory correction changed')
    return corrected


def preflight(root, directory):
    bound = binding(root); checked = read(previous.DIRECTORY/'checked-state.json')
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted implementation changed')
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch86 parent85 and longest reference operands verified; historical input gates retained')
    save(directory/'input-binding.json', bound); save(directory/'dependency-inventory.json', inventory(root))
    save(directory/'supplement-decision.json', supplements())
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    for name, source in (('price', OPERAND), ('calendar', CALENDAR)):
        shutil.copyfile(source, directory/(name+'-input.parquet'))
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False); rows = []
    for number, (name, sha) in enumerate(zip(SOURCES, SOURCE_SHA, strict=True)):
        path = Path('repo/量化策略源代码')/name; text, encoding = read_source(path)
        require(file_sha(path) == sha, 'Source changed')
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(path, copied)
        code = adapted_source(text, number); adapted = folder/(strategy_id(name)+'.adapted.py')
        with adapted.open('x', encoding='utf-8', newline='\n') as out: out.write(code)
        rows.append({'source_path': str(path), 'source_copy': str(copied), 'source_sha256': sha,
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()), 'code_start_line': 9,
            'adapted_copy': str(adapted), 'adapted_sha256': file_sha(adapted),
            'adaptation': 'fix_print only; no imports or economic changes' if number == 0 else 'preface padding only',
            'definition_ast': definitions(ast.parse(code))})
    save(directory/'sources-frozen.json', rows)
    return {'status': 'ok', 'sources_frozen': len(rows), 'not_a_backtest': True}


def start(root, directory):
    binding(root, directory); prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}; rows = read(directory/'sources-frozen.json'); refs = []
    for name, row in zip(SOURCES, rows, strict=True):
        require(old[name]['review_status'] != '人工审查完成' and old[name]['bytes_sha256'] == row['source_sha256'], 'Source reviewed/changed')
        row['review'] = REVIEWS[name]; row['newly_reviewed'] = True
    (directory/'references').mkdir(exist_ok=False)
    for repo, name in REFERENCES:
        path = Path(repo)/name; copied = directory/'references'/path.name; shutil.copyfile(path, copied)
        refs.append({'file': str(path), 'copy': str(copied), 'sha256': file_sha(path),
            'commit': subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
            'use': 'Existing moving-average/native flow/incremental-order reference; no alternate ledger or invented wizard helper'})
    catalog = catalog_strategies(root, 'repo/量化策略源代码'); current = read(Path(catalog['output'])/'catalog.json')
    require(len(old) == len(current) == 695 and {r['path'] for r in current if r != old[r['path']]} == set(SOURCES), 'Unrelated catalog changed')
    save(directory/'source-reviews/review.json', {'sources': rows, 'references': refs, 'catalog': catalog,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior), 'snapshot': SNAPSHOT, 'not_a_backtest': True})
    return archive(root, 'Batch86 full volume-amplitude and technical-flow sources frozen; original trade inputs blocked')


def validate_sources(directory):
    doc = read(directory/'source-reviews/review.json'); frozen = read(directory/'sources-frozen.json')
    require(doc['snapshot'] == SNAPSHOT and len(doc['sources']) == len(frozen) == 2 and
        file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Review binding changed')
    for number, (name, sha, row, original) in enumerate(zip(SOURCES, SOURCE_SHA, doc['sources'], frozen, strict=True)):
        text, encoding = read_source(Path(row['source_copy'])); code = adapted_source(text, number)
        require({k: row[k] for k in original} == original and file_sha(row['source_path']) == file_sha(row['source_copy']) == sha and
            row['review'] == REVIEWS[name] and encoding == row['encoding'] and len(text.splitlines()) == row['complete_lines_read'] and
            Path(row['adapted_copy']).read_text(encoding='utf-8') == code and file_sha(row['adapted_copy']) == row['adapted_sha256'] and
            definitions(ast.parse(code)) == row['definition_ast'], 'Source/AST changed')
    require(len(doc['references']) == len(REFERENCES), 'References missing')
    for (repo, name), row in zip(REFERENCES, doc['references'], strict=True):
        require(row['file'] == str(Path(repo)/name) and file_sha(row['file']) == file_sha(row['copy']) == row['sha256'], 'Reference changed')
    require(supplements() == read(directory/'supplement-decision.json'), 'Supplement binding changed')


def context54(directory, positions=None):
    context = SimpleNamespace(current_dt=dt.datetime(2020, 1, 6, 9, 30),
        portfolio=SimpleNamespace(positions=dict(positions or {}), available_cash=10000., total_value=100000.))
    events = []; ns = {'g': SimpleNamespace(), 'log': SimpleNamespace(info=lambda *a: None),
        'set_benchmark': lambda v: events.append(['benchmark', v]), 'set_option': lambda *a: events.append(['option', *a]),
        'OrderCost': lambda **k: k, 'set_order_cost': lambda c, **k: events.append(['cost', c, k]),
        'run_daily': lambda f, **k: events.append(['schedule', f.__name__, k]),
        'get_current_data': lambda: {'s': SimpleNamespace(paused=False, name='fixture', last_price=10.)}}
    selected(directory, 0, [r['name'] for r in definitions(tree_for(directory, 0))], ns)
    ns['initialize'](context)
    return ns, context, events


def fixture_history(n=32, volume=100.):
    h = {k: np.full(n, 10., dtype=float) for k in ('open', 'close', 'high', 'low')}
    h['volume'] = np.full(n, volume, dtype=float)
    if n: h['volume'][0] = 1000.; h['volume'][-1] = 250.
    return h


def buy_reference(h):
    """Independent chronological interval search; no negative-index state machine."""
    n = len(h['volume']); yesterday = n-1
    if h['close'][yesterday] < h['open'][yesterday]: return False
    candidates = [j for j in range(yesterday) if h['volume'][j] > .8*h['volume'][yesterday]]
    if not candidates: return False
    begin = max(candidates)+1
    if yesterday-begin < 30: return False
    avg = math.fsum(float(x) for x in h['volume'][begin:yesterday])/(yesterday-begin)
    avg5 = math.fsum(float(x) for x in h['volume'][-6:-1])/len(h['volume'][-6:-1])
    if h['volume'][yesterday] < 2.5*avg or h['volume'][yesterday] < 2.5*avg5: return False
    return (max(h['high'][begin:yesterday])-min(h['low'][begin:yesterday]))/h['open'][begin] <= .1


def diagnostics(directory):
    rows = []; ns, ctx, events = context54(directory)
    rows.append({'case': '54_defaults', 'state': vars(ns['g']), 'events': events})
    for label, n, change in [('exact30', 32, {}), ('only29', 31, {}), ('missing_anchor', 32, {'anchor': 0.}),
            ('red_close', 32, {'close': 9.}), ('doji', 32, {'close': 10.}),
            ('volume_below', 32, {'last_volume': 249.}), ('volume_above', 32, {'last_volume': 251.}),
            ('amplitude_equal', 32, {'high': 11.}), ('amplitude_above', 32, {'high': 11.01}),
            ('strict_anchor_equal', 32, {'anchor': 200.}), ('strict_anchor_above', 32, {'anchor': 200.01})]:
        h = fixture_history(n)
        if 'anchor' in change: h['volume'][0] = change['anchor']
        if 'last_volume' in change: h['volume'][-1] = change['last_volume']
        if 'close' in change: h['close'][-1] = change['close']
        if 'high' in change: h['high'][1:-1] = change['high']
        ns['attribute_history'] = lambda *a, h=h, **k: h; ns['mean'] = np.mean
        original = bool(ns['buy_check'](ctx, 's')); independent = bool(buy_reference(h))
        require(original == independent, 'Synthetic buy reference differs')
        rows.append({'case': '54_buy_'+label, 'original': original, 'independent': independent, 'history_rows': n, 'fixture_only': True})
    for label, name, expected in [('missing_mean', 'mean', NameError), ('empty_history', None, IndexError)]:
        ns, ctx, _ = context54(directory); ns['attribute_history'] = lambda *a, **k: fixture_history(0 if name is None else 32)
        if name is None: ns['mean'] = np.mean
        try: ns['buy_check'](ctx, 's')
        except expected as exc: rows.append({'case': '54_'+label, 'error': type(exc).__name__})
        else: raise ValueError('Expected original boundary failure')
    ns, ctx, _ = context54(directory)
    try: ns['sell_check_turnover_ratio'](ctx, 's')
    except NameError as exc: rows.append({'case': '54_missing_timedelta', 'error': str(exc)})
    else: raise ValueError('Original timedelta unexpectedly available')
    for threshold in (.1, .2):
        trace = []; ns['sell_check_mean_price'] = lambda *a: trace.append(a[-1]) or a[-1] == threshold
        ns['sell_check_turnover_ratio'] = lambda *a: trace.append('turnover') or False
        result = ns['sell_check'](ctx, 's')
        rows.append({'case': '54_sell_branch_'+str(threshold), 'result': result, 'trace': trace, 'fixture_only': True})
    for cash, count in [(4999., 0), (5000., 0), (10000., 10), (10000., 11)]:
        ns, ctx, _ = context54(directory, {str(i): 1 for i in range(count)}); ctx.portfolio.available_cash = cash
        ns['g'].sell_list = []; ns['g'].buy_list = ['s']; orders = []; ns['order_target'] = lambda *a: orders.append(list(a))
        with redirect_stdout(StringIO()): ns['market_open'](ctx)
        rows.append({'case': f'54_cash_{cash}_positions_{count}', 'orders': orders, 'fixture_only': True})
    ns, ctx, _ = context54(directory, {'s': SimpleNamespace(value=20000., price=10., total_amount=2000)})
    ns['g'].sell_list = []; ns['g'].buy_list = ['s']; orders = []; ns['order_target'] = lambda *a: orders.append(list(a))
    ns['market_open'](ctx); rows.append({'case': '54_add_comment_targets_lower_quantity', 'orders': orders, 'held_quantity': 2000})
    ns, ctx, _ = context54(directory); sent = []; ns['send_message'] = lambda *a: sent.append(True)
    try: ns['send_wx_msg'](ctx, ['s'], [])
    except TypeError as exc: rows.append({'case': '54_message_bytes_type', 'error': type(exc).__name__, 'messages_sent': len(sent)})
    else: raise ValueError('Expected bytes/text failure')
    # Freeze the actual natural-day argument with an inert query object, never a remote API.
    ns, ctx, _ = context54(directory); trace = []
    class Field:
        def __eq__(self, value): return ('code', value)
        def __gt__(self, value): return ('gt', value)
    ns.update(timedelta=dt.timedelta, valuation=SimpleNamespace(code=Field(), turnover_ratio=Field()),
        query=lambda *a: SimpleNamespace(filter=lambda *filters: filters),
        get_fundamentals=lambda q, day: trace.append([q, day]) or pd.DataFrame())
    ns['sell_check_turnover_ratio'](ctx, 's'); rows.append({'case': '54_turnover_previous_sunday', 'trace': trace, 'fixture_only': True})
    ns = {}; ctx = SimpleNamespace(current_dt=dt.datetime(2020, 1, 6),
        portfolio=SimpleNamespace(positions={'held': 1}, available_cash=1000., total_value=2000.))
    events = []; ns.update(g=SimpleNamespace(), pd=pd,
        log=SimpleNamespace(set_level=lambda *a: None), set_benchmark=lambda v: events.append(['benchmark', v]),
        FixedSlippage=lambda v: v, set_slippage=lambda v: events.append(['slippage', v]),
        set_option=lambda *a: events.append(['option', *a]), OrderCost=lambda **k: k,
        set_order_cost=lambda c, **k: events.append(['cost', c, k]), run_daily=lambda f, when: events.append(['schedule', f.__name__, when]),
        valuation=SimpleNamespace(circulating_market_cap='cap', pe_ratio='pe'), indicator=SimpleNamespace(roe='roe'))
    selected(directory, 1, [r['name'] for r in definitions(tree_for(directory, 1))], ns); ns['initialize'](ctx)
    rows.append({'case': '95_defaults', 'state': dict(vars(ns['g'])), 'events': events})
    trace = []
    for fn in ('MACD_judge_jincha', 'BBands_lowerband_judge_dayu', 'MA_judge_duotou'):
        ns[fn] = lambda *a, fn=fn: trace.append([fn, *a]) or True
    ns['technical_indicators_filter'](ctx, ['a']); rows.append({'case': '95_technical_arguments', 'trace': list(trace), 'fixture_only': True})
    trace.clear(); ns['money_flow_dayu'] = lambda *a, **k: trace.append([a[1:], k]) or True
    ns['other_func_filter'](ctx, ['a']); ns['g'].check_out_lists = ['a']
    for fn in ('st_filter', 'paused_filter', 'delisted_filter', 'high_limit_filter'): ns[fn] = lambda c, s: s
    ns['sell'] = lambda c, s: trace.append(['sell', s]); ns['buy'] = lambda c, s: trace.append(['buy', s])
    ns['trade'](ctx); rows.append({'case': '95_flow_arguments_and_order', 'trace': list(trace), 'fixture_only': True})
    ns = selected(directory, 1, ['get_check_stocks_sort_input_dict', 'check_stocks_sort', 'sell_every_day', 'holded_filter'],
        {'g': SimpleNamespace(open_sell_securities=['held', 'held', 'absent'], filter_holded=False), 'pd': pd,
         'valuation': SimpleNamespace(circulating_market_cap='cap', pe_ratio='pe'), 'indicator': SimpleNamespace(roe='roe')})
    weights = ns['get_check_stocks_sort_input_dict'](); trace = []
    ns['get_sort_dataframe'] = lambda s, field, weight: trace.append([field, weight]) or pd.DataFrame({field: [0.]}, index=['a'])
    try: ns['check_stocks_sort'](ctx, ['a'], weights)
    except AttributeError as exc: rows.append({'case': '95_nonempty_zero_weight_sort', 'weights': weights, 'calls': trace, 'error': str(exc), 'fixture_only': True})
    else: raise ValueError('Expected obsolete DataFrame.sort failure')
    trace = []; ns['order_target_value'] = lambda *a: trace.append(list(a))
    ns['sell_every_day'](ctx); rows.append({'case': '95_retry_no_cleanup', 'orders': trace, 'retry': ns['g'].open_sell_securities})
    rows.append({'case': '95_holded_false', 'result': ns['holded_filter'](ctx, ['held', 'new', 'new'])})
    return {'cases': plain(rows), 'case_count': len(rows), 'not_a_backtest': True,
        'limits': ['np.mean is a declared arithmetic reference injection, not proof of original platform mean',
            'Synthetic state/history/helper returns do not admit original historical pool, signals, fills or messaging']}


def validate_price_shape(price):
    require(len(price) == 46781 and not price.duplicated(['instrument', 'date']).any(), 'Reference input shape changed')


def component(directory):
    price = pd.read_parquet(directory/'price-input.parquet'); calendar = pd.read_parquet(directory/'calendar-input.parquet')
    validate_price_shape(price)
    days = calendar.loc[calendar.is_open, 'date'].tolist(); rows = []; excluded = []; profiles = []; max_ma_error = 0.
    ns, ctx, _ = context54(directory); ns['mean'] = np.mean
    for stock, group in price.groupby('instrument', sort=True):
        first = group.date.min(); series = group.set_index('date').reindex([d for d in days if d >= first]); valid_history = []
        counts = {'buy_true': 0, 'ma10_true': 0, 'ma20_true': 0, 'buy_windows': 0, 'ma_windows': 0, 'partial_buy_windows': 0}
        for day, current in series.iterrows():
            if pd.isna(current.is_trading):
                valid_history.append(None); excluded.append({'instrument': stock, 'date': day, 'reason': 'missing_session'}); continue
            if not bool(current.is_trading): continue
            prior = valid_history[-270:]
            reason = 'missing_history' if not prior else ('unverified_history' if any(x is None for x in prior) else None)
            if reason is None:
                window = pd.DataFrame(prior)
                values = window[['open_adj', 'close_adj', 'high_adj', 'low_adj', 'volume']].to_numpy(float)
                factor = float(current.back_factor)
                if (not np.isfinite(values).all() or not math.isfinite(factor) or factor <= 0 or
                        window.adjustment_status.ne('usable').any() or current.adjustment_status != 'usable' or
                        (values[:, :4] <= 0).any() or (values[:, 4] < 0).any()): reason = 'unusable_price_volume'
            if reason is not None: excluded.append({'instrument': stock, 'date': day, 'reason': reason})
            else:
                h = {k: window[k+'_adj'].to_numpy(float)/factor for k in ('open', 'close', 'high', 'low')}
                h['volume'] = window.volume.to_numpy(float)
                ns['attribute_history'] = lambda *a, **k: h
                original = bool(ns['buy_check'](ctx, 's')); independent = bool(buy_reference(h))
                require(original == independent, f'{stock}/{day}: buy threshold difference')
                counts['buy_windows'] += 1; counts['partial_buy_windows'] += int(len(window) < 270); counts['buy_true'] += int(original)
                ma10 = ma20 = diff = original_diff = None
                if len(window) >= 31:
                    ns['attribute_history'] = lambda *a, **k: {key: value[-31:] for key, value in h.items()}
                    ctx.portfolio.positions = {'s': 1}
                    ma10 = bool(ns['sell_check_mean_price'](ctx, 's', .1)); ma20 = bool(ns['sell_check_mean_price'](ctx, 's', .2))
                    mean5 = math.fsum(float(x) for x in h['close'][-5:])/5
                    mean30 = math.fsum(float(x) for x in h['close'][-30:])/30
                    diff = (mean5-mean30)/mean30
                    original_mean5 = float(np.mean(h['close'][-5:])); original_mean30 = float(np.mean(h['close'][-30:]))
                    original_diff = (original_mean5-original_mean30)/original_mean30
                    max_ma_error = max(max_ma_error, abs(diff-original_diff))
                    require(math.isclose(diff, original_diff, rel_tol=1e-10, abs_tol=1e-12), f'{stock}/{day}: MA numeric difference')
                    require(ma10 == (diff > .1) and ma20 == (diff > .2), f'{stock}/{day}: MA threshold difference')
                    counts['ma_windows'] += 1; counts['ma10_true'] += int(ma10); counts['ma20_true'] += int(ma20)
                rows.append({'instrument': stock, 'decision_date': day, 'last_history_date': prior[-1]['date'],
                    'history_rows': len(window), 'buy_reference': original, 'ma10_reference': ma10, 'ma20_reference': ma20,
                    'ma5_ma30_independent_diff': diff, 'ma5_ma30_numpy_reference_diff': original_diff})
            valid_history.append(dict(current, date=day))
        profiles.append({'instrument': stock, 'first': first, 'last': group.date.max(), **counts})
    frame = pd.DataFrame(rows)
    result = {'input_rows': len(price), 'input_sha256': OPERAND_SHA, 'calendar_sha256': CALENDAR_SHA,
        'first_decision': frame.decision_date.min(), 'last_decision': frame.decision_date.max(), 'profiles': profiles,
        'buy_reference_windows': len(frame), 'ma_reference_windows': int(frame.ma10_reference.notna().sum()),
        'buy_reference_true': int(frame.buy_reference.sum()), 'partial_buy_windows': int(frame.history_rows.lt(270).sum()),
        'excluded': excluded, 'independent_boolean_differences': 0, 'max_MA_difference_absolute_error': max_ma_error,
        'original_complete_pool_signal_days': 0, 'original95_technical_signal_windows': 0,
        'original95_flow_windows': 0, 'new_trading_backtests': 0, 'cost_backtests': [], 'not_a_backtest': True,
        'numpy': np.__version__, 'pandas': pd.__version__,
        'reference_policy': 'Previous observed trading rows only; explicitly paused rows skipped, missing sessions blocked, max270 including partial original history; price anchored by decision-day factor; volume unchanged. MA reference requires31 prior rows.',
        'limits': ['Arithmetic uses explicit numpy.mean injection and independently math.fsum; platform mean/volume/real-price equivalence unproved',
            'Reference decision-day factor basis may differ from platform history; no complete historical universe or turnover admission',
            'Original money field requested but unread by the selected kernels; no fabricated money provided',
            'Full95 technical helpers and main-flow history unavailable; no default TA-Lib substitution or invented rank/fills/returns']}
    return plain(result), frame


def study(root, directory):
    binding(root, directory); validate_sources(directory); validate_inventory(root, directory); before = implementation()
    diag = diagnostics(directory); summary, frame = component(directory)
    require(before == implementation(), 'Implementation changed during study')
    save(directory/'diagnostics.json', diag); save(directory/'component-research.json', summary)
    with (directory/'component-values.parquet').open('xb') as out: frame.to_parquet(out, index=False)
    return archive(root, 'Batch86 longest volume-amplitude/MA arithmetic and original compatibility diagnostics archived; no trades admitted')


def offline(root, directory):
    binding(root, directory); validate_sources(directory); before = implementation()
    def forbidden(*a, **k): raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), patch.object(requests.sessions.Session, 'request', forbidden):
        diag = diagnostics(directory); summary, frame = component(directory); dep = validate_inventory(root, directory); catalog = offline_catalog(root, directory)
    require(diag == read(directory/'diagnostics.json') and summary == read(directory/'component-research.json') and
        dep == read(directory/'dependency-inventory-corrected.json'), 'Offline JSON evidence differs')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(directory/'component-values.parquet'), check_exact=True)
    require(before == implementation(), 'Implementation changed during offline')
    save(directory/'offline-catalog.json', catalog); validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'result': 'match', 'differences': 0, 'socket_network_disabled': True,
        'implementation_sha256': before, 'catalog': catalog, 'component_rows': len(frame), 'new_trading_backtests': 0, 'not_a_backtest': True})
    return archive(root, 'Batch86 forbidden-network literal diagnostics/long reference arithmetic/inventory and three catalogs match0')


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/review_strategy_batch86.py'],
        [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch86_research.py',
            'tests/unit/test_batch85_research.py', 'tests/unit/test_catalog_dependencies.py', 'tests/unit/test_catalog_schedules.py',
            'tests/unit/test_research_metadata.py', 'tests/integration/test_strategy_merge.py'], ['git', 'diff', '--check']]


def checks(root, directory):
    binding(root, directory); validate_sources(directory); validate_catalog(root, directory); before = implementation(); rows = []
    require(CORE <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}, 'Core missing')
    folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for k, command in enumerate(commands()):
        path = folder/f'check-{k}.log'
        with path.open('x') as out: process = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        rows.append({'command': command, 'returncode': process.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(process.returncode == 0, 'Check failed: '+str(path))
    require(before == implementation(), 'Implementation changed during checks')
    for name in FILES:
        path = directory/'implementation-checked'/name; path.parent.mkdir(parents=True, exist_ok=True); require(not path.exists(), 'Checked copy exists'); shutil.copyfile(name, path)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': rows, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed'}


def finish(root, directory):
    binding(root, directory); validate_sources(directory); validate_inventory(root, directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json'); current = implementation()
    require(checked['status'] == 'passed' and current == checked['implementation_sha256'] == off['implementation_sha256'] and
        CORE <= checked['evidence_sha256'].keys() and [r['command'] for r in checked['commands']] == commands(), 'Checks changed')
    for row in checked['commands']: require(row['returncode'] == 0 and file_sha(row['log']) == row['sha256'], 'Log changed')
    for name, sha in checked['evidence_sha256'].items(): require(file_sha(directory/name) == sha, 'Evidence changed')
    for name, sha in current.items(): require(file_sha(directory/'implementation-checked'/name) == sha, 'Checked copy changed')
    require(off['result'] == 'match' and off['differences'] == 0 and off['socket_network_disabled'] and
        Store(root).published() == read(directory/'baseline.json')['published'], 'Offline/publication changed')
    require(read(directory/'component-research.json')['independent_boolean_differences'] == 0, 'Arithmetic differences remain')
    protection = protect(root, directory); progress = archive(root, 'Batch86 literal volume/MA and wizard-flow source research accepted; trades blocked')
    receipt = Path('docs/handoff/2026-10-09-batch86-verification.json')
    save(receipt, {'status': 'ok', 'snapshot': SNAPSHOT, 'reviews': read(directory/'source-reviews/review.json'),
        'checks': checked, 'offline': off, 'diagnostics': read(directory/'diagnostics.json'),
        'research': read(directory/'component-research.json'), 'supplement': read(directory/'supplement-decision.json'),
        'inventory': read(directory/'dependency-inventory-corrected.json'),
        'inventory_correction': read(directory/'inventory-row-correction.json'), 'protection': protection, 'progress': progress,
        'newly_reviewed': 2, 'not_a_backtest': True, 'new_trading_backtests': 0, 'original_strategy_complete': False})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(receipt),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'final_script_sha256': file_sha(__file__),
        'catalog_sha256': progress['catalog_sha256']})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'newly_reviewed': 2, 'new_trading_backtests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'correct_inventory', 'study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args(); print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str))
