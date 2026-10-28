"""Freeze intraday source research; reference MA predicates never become trade signals."""
import argparse
import ast
from contextlib import redirect_stdout
import datetime as dt
from io import StringIO
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source, strategy_id
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch61 import OPERAND, OPERAND_SHA, CALENDAR, CALENDAR_SHA
from scripts.review_strategy_batch83 import definitions, plain, selected_nodes
from scripts.run_strategy_batch4 import checkpoint, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

DIRECTORY = Path('data/staging/strategies-batch90/20261009-limit-near-high-shortline-stage')
SNAPSHOT = '20261006-145049-e06c'
SOURCES = ('2023年度精选策略/89.2019年到现在90多倍的新策略.txt',
           '2024年度精选策略2/8.14个月200%，超短线实盘交易策略！无未来函数.txt')
SOURCE_SHA = ('1d132c408b37fdfffcaafe849c3a2d31376cd7480e28c5c68863b6cfb17f1c6f',
              '32eff6b4566c5ef23bd350725f0e4b9543034dfb110fdde779153046a58df266')


def binding(directory=None):
    review = read('docs/handoff/2026-10-09-batch89-final-review.json')
    for name in ('receipt', 'checked_state', 'final_binding'):
        require(file_sha(review[name+'_file']) == review[name+'_sha256'], 'Accepted89 binding changed')
    checked = read(review['checked_state_file'])
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted code changed')
    parent = Path(review['accepted_directory'])
    require(all(file_sha(parent/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Accepted evidence changed')
    prior = read('docs/handoff/2026-10-07-batch47-verification.json')['checks']['evidence_sha256']
    require(file_sha(OPERAND) == OPERAND_SHA == prior['price-input.parquet'] and
            file_sha(CALENDAR) == CALENDAR_SHA == prior['calendar-input.parquet'], 'Reference operands changed')
    result = {'snapshot': SNAPSHOT, 'accepted_checkpoint': review['accepted_checkpoint'],
        'parent_review_sha256': file_sha('docs/handoff/2026-10-09-batch89-final-review.json'),
        'price_file': str(OPERAND), 'price_sha256': OPERAND_SHA,
        'calendar_file': str(CALENDAR), 'calendar_sha256': CALENDAR_SHA,
        'formal_acceptance': False, 'not_a_backtest': True}
    if directory is not None:
        require(read(directory/'input-binding.json') == result, 'Stage binding changed')
        for name in ('price', 'calendar'):
            require(file_sha(directory/(name+'-input.parquet')) == result[name+'_sha256'], 'Stage operand changed')
    return result


def tree(number, directory=None):
    path = Path('repo/量化策略源代码')/SOURCES[number]
    require(file_sha(path) == SOURCE_SHA[number], 'Original source changed')
    if directory is not None:
        row = read(directory/'sources-frozen.json')[number]
        require(row['source_path'] == str(path) and file_sha(row['source_copy']) == SOURCE_SHA[number], 'Frozen source changed')
    return ast.parse(read_source(path)[0])


def namespace(number, values=None, directory=None):
    parsed = tree(number, directory)
    return selected_nodes(parsed, [n['name'] for n in definitions(parsed)], dict(values or {}))


def stage(root, directory):
    bound = binding(); store = Store(root); state = store.state(SNAPSHOT)
    require(store.published()['batch_id'] == state['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    catalog = read(Path(read(Path(root)/'catalog/strategies/latest.json')['directory'])/'catalog.json')
    for number, name in enumerate(SOURCES):
        tree(number)
        row = next(r for r in catalog if r['path'] == name)
        require(row['review_status'] != '人工审查完成' and row['bytes_sha256'] == SOURCE_SHA[number], 'Review scope changed')
    checkpoint(root, directory, 'Batch90 original intraday sources start; no original pool/minute/flow admitted')
    save(directory/'input-binding.json', bound)
    for name in ('price', 'calendar'):
        shutil.copyfile(bound[name+'_file'], directory/(name+'-input.parquet'))
    rows = []; folder = directory/'sources'; folder.mkdir(exist_ok=False)
    for number, name in enumerate(SOURCES):
        source = Path('repo/量化策略源代码')/name; text, encoding = read_source(source)
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(source, copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'sha256': SOURCE_SHA[number],
            'encoding': encoding, 'complete_lines_read': len(text.splitlines()),
            'definitions': definitions(tree(number)), 'top_level_executed': False})
    save(directory/'sources-frozen.json', rows)
    registered = {t: sum(e['rows'] for e in entries.values()) for t, entries in state['tables'].items()}
    price = pd.read_parquet(OPERAND)
    save(directory/'dependency-inventory.json', {'snapshot': SNAPSHOT, 'registered_table_rows': registered,
        'reference_rows': len(price), 'reference_instruments': sorted(price.instrument.unique()),
        'reference_range': plain([price.date.min(), price.date.max()]),
        'original_1m_registered': 'bars_1m' in state['tables'],
        'money_flow_registered': any('money_flow' in name for name in state['tables']),
        'original_signal_rows_admitted': 0, 'not_a_backtest': True,
        'limits': ['No admitted historical all-stock pool/name/ST chain or original complete1m/tick rows',
            'Ten frozen stocks are a price-volume reference, not original pool or full candidate ranking',
            '89 needs original Panel fields pre_close/high_limit and dynamic index/stock prices',
            '8 needs original MACD seed/window and money_flow.change_pct history/availability',
            'Skip-paused, volume adjustment, fees selected as fund for stock, immediate fills and zero-position keys unproved']})
    previous = Path('data/staging/strategies-batch82/20261009-intraday-reversal-price-volume')
    receipt = read('docs/handoff/2026-10-09-batch82-verification.json')
    links = {}
    for name in ('existing-apis.json', 'probe-results.json'):
        path = previous/name
        require(file_sha(path) == receipt['checks']['evidence_sha256'][name], 'Minute evidence changed')
        links[name] = {'file': str(path), 'sha256': file_sha(path)}
    save(directory/'supplement-decision.json', {'previous_minute_evidence': links,
        'new_requests': 0, 'new_rows': 0, 'published': False, 'not_a_backtest': True,
        'decision': 'Reuse verified1m fixed-ndays5 and DNS failure; no same-host blind retries or latest pool/flow substitution',
        'restore_required': ['Historical all-stock pool/name/status and platform Panel/limit/adjustment semantics',
            '89 completed1m/index/stock dynamic prices and 8 MACD/flow vintage/open fields',
            'UniqueBook intraday execution with original order sequencing and stock fee evidence']})
    return {'status': 'stage_frozen', 'sources': len(rows), 'formal_acceptance': False, 'not_a_backtest': True}


def diagnostics(directory=None):
    cases = []; events = []; ns = namespace(0, directory=directory)
    ns['get_current_data'] = lambda: {'000300.XSHG': SimpleNamespace(last_price=101., day_open=100.)}
    ns['toBuy'] = lambda context: events.append('buy'); ns['toSell'] = lambda context: events.append('sell')
    for hhmm, expected in ((930, []), (931, ['buy']), (1030, ['buy']), (1031, []), (1450, ['sell']), (1451, [])):
        events.clear(); context = SimpleNamespace(current_dt=dt.datetime(2021, 1, 4, hhmm//100, hhmm%100))
        ns['handle_data'](context, {})
        require(events == expected, 'Original callback time differs')
        cases.append({'case': f'89_time_{hhmm}', 'events': list(events), 'expected': expected})
    ns['get_current_data'] = lambda: {'000300.XSHG': SimpleNamespace(last_price=100., day_open=100.)}
    events.clear(); ns['handle_data'](SimpleNamespace(current_dt=dt.datetime(2021, 1, 4, 9, 31)), {})
    require(events == [], 'Index equality passed strict entry')
    cases.append({'case': '89_index_equal_open', 'events': [], 'expected': []})
    for last, expected in ((109., []), (109.01, [['s', 10000.]])):
        orders = []; ns = namespace(0, {'g': SimpleNamespace(pool=['s'], max_holding=2),
            'get_current_data': lambda last=last: {'s': SimpleNamespace(last_price=last, name='fixture')},
            'attribute_history': lambda *a: {'close': np.array([100.])},
            'order_target_value': lambda *a: orders.append(list(a)), 'log': SimpleNamespace(info=lambda *a: None)}, directory)
        context = SimpleNamespace(portfolio=SimpleNamespace(positions={}, available_cash=20000.))
        ns['toBuy'](context); require(orders == expected, 'Strict1.09 entry differs')
        cases.append({'case': f'89_stock_price_{last}', 'orders': orders, 'expected': expected,
            'array_shape_synthetic_not_proven_platform': True})
    ns['attribute_history'] = lambda *a: {'close': pd.Series([100.], index=pd.to_datetime(['2021-01-01']))}
    try:
        ns['toBuy'](context)
    except KeyError as exc:
        require(exc.args == (-1,), 'Unexpected Series compatibility error')
        cases.append({'case': '89_datetime_series_minus1', 'exception': 'KeyError', 'label': -1})
    else:
        raise ValueError('Expected pandas label failure')
    ns = namespace(1, {'datetime': dt}, directory)
    stocks = ['600000.XSHG', '300001.XSHE', '688001.XSHG', '600001.XSHG']
    for index, names, expected in ((stocks, ['ST name', 'normal', 'normal', '*退 name'], [stocks[0], stocks[3]]),
        (['ST', '600000.XSHG'], ['normal', 'normal'], [])):
        pool = pd.DataFrame({'display_name': names}, index=index)
        actual = ns['filter_stock'](pool); require(actual == expected, 'Original Series membership differs')
        cases.append({'case': '8_filter_'+str(len(cases)), 'pool': index, 'names': names,
            'actual': actual, 'expected': expected, 'membership_checks_series_index': True})
    actual = ns['getday3'](dt.date(2021, 1, 4), -1)
    require(actual == dt.date(2021, 1, 3), 'Natural-day endpoint changed')
    cases.append({'case': '8_natural_day_endpoint', 'actual': str(actual),
        'expected': '2021-01-03', 'not_a_proven_trading_endpoint_difference': True})
    require('getday' in ns and 'buy_stock' in ns, 'Definitions missing')
    return {'cases': cases, 'case_count': len(cases), 'synthetic_only': True, 'not_a_backtest': True,
        'platform_equivalence_proven': False, 'new_trading_backtests': 0}


def ma_predicate(directory=None):
    parent = next(n for n in tree(1, directory).body if isinstance(n, ast.FunctionDef) and n.name == 'before_trading_start')
    loop = next(n for n in parent.body if isinstance(n, ast.For) and n.target.id == 'stock')
    node = next(n for n in loop.body if isinstance(n, ast.If))
    require(not any(isinstance(n, (ast.Call, ast.Attribute, ast.Subscript)) for n in ast.walk(node.test)), 'Predicate scope changed')
    return compile(ast.Expression(body=node.test), '<batch90-pinned-six-MA-conditions>', 'eval')


def component(directory):
    binding(directory); tree(0, directory); code = ma_predicate(directory)
    frame = pd.read_parquet(directory/'price-input.parquet'); rows = []; disagreements = []
    for instrument, data in frame.groupby('instrument', sort=True):
        data = data.loc[data.is_trading].sort_values('date').reset_index(drop=True)
        require(data.adjustment_status.eq('usable').all(), 'Unusable reference adjustment')
        close = data.close_adj.to_numpy(float); volume = data.volume.to_numpy(float)
        for k in range(30, len(data)):
            local = {}; independent = {}
            for shift, label in ((0, 'p1'), (1, 'p2')):
                for n in (5, 10, 20, 30):
                    for name, values in (('close', close), ('volume', volume)):
                        window = values[k-shift-n+1:k-shift+1]
                        key = f'{label}_ma{n}_{name}'
                        local[key] = float(np.mean(window)); independent[key] = math.fsum(map(float, window))/n
            actual = bool(eval(code, {'__builtins__': {}}, local))
            x = independent
            expected = bool(x['p1_ma5_close'] < x['p1_ma10_close'] and x['p2_ma5_close'] > x['p2_ma10_close'] and
                x['p2_ma5_volume'] > x['p2_ma10_volume'] and x['p1_ma5_volume']*1.2 > x['p1_ma10_volume'] and
                x['p1_ma20_close'] > x['p2_ma20_close'] and x['p1_ma30_close'] > x['p2_ma30_close'])
            row = {'instrument': instrument, 'date': data.date.iloc[k], 'original_literal_predicate': actual,
                'independent_fsum_predicate': expected, **local}
            rows.append(row)
            if actual != expected:
                disagreements.append({'instrument': instrument, 'date': plain(data.date.iloc[k]), 'numpy': actual, 'fsum': expected})
    return pd.DataFrame(rows), {'reference_windows': len(rows), 'predicate_true': sum(r['original_literal_predicate'] for r in rows),
        'independent_condition_differences': len(disagreements), 'differences': disagreements,
        'range': plain([min(r['date'] for r in rows), max(r['date'] for r in rows)]),
        'original_full_signal_rows': 0, 'new_trading_backtests': 0, 'not_a_backtest': True,
        'limits': ['Exact original six-condition AST evaluated on ten reference stocks, NumPy window means; not original groupby/platform inputs',
            'Fsum is an independent diagnostic only; numerical disagreements retained, no original repair',
            'Verified nonpaused reference rows retained; daily natural-day endpoints have no fabricated extra sessions',
            'Common positive decision-date anchoring cancels in these price comparisons; volume adjustment semantics unproved',
            'MACD, change_pct, low-open filter, full historical pool and minute execution absent; true predicate is not a buy signal']}


def audit(root, directory):
    binding(directory)
    result = network_disabled(lambda: diagnostics(directory))
    values, summary = network_disabled(lambda: component(directory))
    save(directory/'diagnostics.json', result); save(directory/'component-research.json', summary)
    with (directory/'MA-values.parquet').open('xb') as stream:
        values.to_parquet(stream, index=False)
    return {'status': 'stage_audited', 'cases': result['case_count'], **summary}


def offline(root, directory):
    binding(directory)
    require(network_disabled(lambda: diagnostics(directory)) == read(directory/'diagnostics.json'), 'Diagnostic replay differs')
    values, summary = network_disabled(lambda: component(directory))
    require(summary == read(directory/'component-research.json'), 'Summary replay differs')
    pd.testing.assert_frame_equal(values, pd.read_parquet(directory/'MA-values.parquet'), check_exact=True)
    result = {'status': 'match', 'differences': 0, 'socket_HTTP_disabled': True,
        'reference_windows': len(values), 'not_a_backtest': True, 'formal_acceptance': False}
    save(directory/'offline-verification.json', result)
    save(directory/'stage-evidence.json', {'implementation_sha256': file_sha(__file__),
        'files': {p.relative_to(directory).as_posix(): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()},
        'formal_acceptance': False, 'not_a_backtest': True})
    progress = archive(root, 'Batch90 original intraday source stage; six MA conditions on ten-stock reference; no full signals/trades; final checks pending')
    return {**result, 'checkpoint': progress['checkpoint']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'audit', 'offline'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    with redirect_stdout(StringIO()) as captured:
        result = globals()[args.action](args.root, args.directory)
    print(json.dumps(plain(result), ensure_ascii=False))
