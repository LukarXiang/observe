"""Freeze all-weather sources and native input/compatibility evidence before repair."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source, strategy_id
from scripts import verify_strategy_batch88 as previous
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch58 import definitions
from scripts.review_strategy_batch73 import safe_function
from scripts.run_strategy_batch4 import checkpoint, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch89/20261009-all-weather-source-stage')
SNAPSHOT = previous.stage.SNAPSHOT
SOURCES = ('2024年度精选策略2/7.致敬经典作品——小兵哥《一致性风险度量》——极速版.txt',
           '2024年度精选策略2/65.神级策略：桥水全天候策略-7年最大回撤仅3.5%.txt')
SOURCE_SHA = ('80903d8634a4b69ac5dd52836bf6442fdad76dca7ce4b313794629a63fd92db6',
              '32340549346ad2e7ad6e7d7e28ac9e7dd53bad0bfd36352e434bc6cb9946e430')
AST_SHA = '8d1a1734bbf568f2baa8dca4e2a88480f710f126a2b9c370f273651fcbfa03bb'
POOL = ('510300.XSHG', '518880.XSHG', '511010.XSHG', '513100.XSHG')
CANDIDATES = (
    ('data/staging/strategies-batch75/20261008-ETF-info-rotation', 'reused-ETF.json', None),
    ('data/staging/strategies-batch74/20261008-FOF-two-eight', 'sina-fallback/results.json', 'fund-518880'),
    ('data/staging/strategies-batch77/20261009-DCA-index-valuation', 'probe-results.json', 'fund-511010'),
    ('data/staging/strategies-batch78/20261009-core-assets-weighted-momentum', 'probe-results.json', 'fund-513100'))


def binding():
    review = read('docs/handoff/2026-10-09-batch88-final-review.json')
    for name, sha in (('receipt_file', 'receipt_sha256'), ('checked_state_file', 'checked_state_sha256'),
                      ('final_binding_file', 'final_binding_sha256')):
        require(file_sha(review[name]) == review[sha], 'Accepted88 binding changed')
    checked = read(review['checked_state_file'])
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()),
            'Accepted implementation changed')
    require(all(file_sha(previous.DIRECTORY/n) == sha for n, sha in checked['evidence_sha256'].items()),
            'Accepted evidence changed')
    return {'snapshot': SNAPSHOT, 'accepted_checkpoint': review['accepted_checkpoint'],
            'accepted_review_sha256': file_sha('docs/handoff/2026-10-09-batch88-final-review.json'),
            'formal_acceptance': False, 'not_a_backtest': True}


def validate_sources(directory):
    require(read(directory/'input-binding.json') == binding(), 'Parent binding changed')
    rows = read(directory/'sources-frozen.json')
    require(len(rows) == len(SOURCES), 'Source scope changed')
    for name, sha, row in zip(SOURCES, SOURCE_SHA, rows, strict=True):
        require(row['source_path'] == str(Path('repo/量化策略源代码')/name) and
                file_sha(row['source_path']) == file_sha(row['source_copy']) == sha, 'Original source changed')


def candidate_inputs():
    inputs = {}
    for instrument, (folder, name, case) in zip(POOL, CANDIDATES, strict=True):
        folder = Path(folder); registry = folder/name; checked = read(folder/'checked-state.json')
        require(file_sha(registry) == checked['evidence_sha256'][name], 'Candidate registry changed')
        doc = read(registry)
        if case is None:
            require(file_sha(doc['receipt']) == doc['receipt_sha256'] and
                    file_sha(doc['probe']) == doc['probe_sha256'], 'Prior510300 binding changed')
            row = doc['payload']
        else:
            matches = [r for r in doc['results'] if r['case'] == case]
            require(len(matches) == 1 and matches[0]['status'] == 'success', 'Candidate missing')
            row = matches[0]
        require(file_sha(row['file']) == row['sha256'], 'Candidate payload changed')
        frame = pd.read_parquet(row['file'])
        require(len(frame) == row['rows'], 'Candidate row count changed')
        inputs[instrument] = {'registry': str(registry), 'registry_sha256': file_sha(registry),
                             'payload': row, 'strict_usable': False}
    return inputs


def inventory(root):
    store = Store(root); state = store.state(SNAPSHOT)
    published = store.published()
    require(published['batch_id'] == state['batch_id'] and published['tables'] == state['tables'],
            'Publication differs from frozen snapshot')
    instruments = [s.replace('.XSHG', '.SH') for s in POOL]
    tables = {}
    for name in ('bars_1d', 'adj_factors', 'adj_coverage', 'corp_actions', 'instruments', 'minute_universe'):
        if name not in state['tables']:
            tables[name] = {'registered': False}; continue
        frame = store.load_state(state, name, filters=[('instrument', 'in', instruments)])
        tables[name] = {'registered': True, 'rows': len(frame),
                        'by_instrument': {s: int(frame.instrument.eq(s).sum()) for s in instruments}}
    return {'snapshot': SNAPSHOT, 'published_batch': store.published()['batch_id'],
            'original_pool': list(POOL), 'standard_tables': tables,
            'limits': ['Candidate close histories do not prove identity/events/adjustment/state/fees/open execution',
                       '513100 is in original money_fund branch; not a cash fund proxy',
                       'No new requests/publication/trading; historical shares/PCF/names gates remain'],
            'strict_usable': False, 'not_a_backtest': True}


def candidate_coverage(inputs):
    rows = {}; common = None
    for instrument, item in inputs.items():
        row = item['payload']; require(file_sha(row['file']) == row['sha256'], 'Candidate changed')
        frame = pd.read_parquet(row['file']); days = pd.to_datetime(frame.date)
        require(days.is_unique and days.is_monotonic_increasing, 'Candidate dates need review')
        eligible = frame.loc[days <= pd.Timestamp('2026-09-29')].copy()
        close = pd.to_numeric(eligible.close, errors='raise')
        require(np.isfinite(close).all() and (close > 0).all(), 'Invalid candidate close')
        dates = pd.to_datetime(eligible.date); observed = set(dates.dt.strftime('%Y-%m-%d'))
        common = observed if common is None else common & observed
        ratio = close.pct_change(fill_method=None)
        jumps = [{'date': str(dates.iloc[k].date()), 'raw_relative_change': float(ratio.iloc[k])}
                 for k in range(1, len(close)) if abs(ratio.iloc[k]) > .5]
        rows[instrument] = {'rows_in_archive': len(frame), 'rows_through_cutoff': len(eligible),
                            'range': [str(dates.min().date()), str(dates.max().date())],
                            'raw_jumps_over_50pct': jumps, 'trade_usable': False}
    common = sorted(common)
    return {'inputs': rows, 'common_observed_dates': len(common),
            'common_range': [common[0], common[-1]] if common else [],
            'limits': ['Observed-date intersection is not verified exchange-calendar coverage or an ES120 window count',
                       'Raw jumps retained; no split/event/correction inferred',
                       'No strategy signals or weights calculated pending return-semantics decision'],
            'not_a_backtest': True}


def original_calc(directory, history):
    path = Path(read(directory/'sources-frozen.json')[0]['source_copy'])
    require(file_sha(path) == SOURCE_SHA[0], 'Unreviewed source')
    tree = ast.parse(read_source(path)[0])
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fun_calc_trade_ratio')
    require(safe_function(node), 'Unsafe source definition')
    namespace = {'history': history, 'math': math, 'log': SimpleNamespace(info=lambda *a: None),
        'g': SimpleNamespace(equity=[POOL[0]], commodities=[POOL[1]], bonds=[POOL[2]],
                             money_fund=[POOL[3]], confidence_level=2.58)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<sha-pinned-batch89-original-calc>', 'exec'), namespace)
    return namespace['fun_calc_trade_ratio']


def compatibility(directory):
    frame = pd.DataFrame({'close': 100+np.sin(np.arange(120.))},
                         index=pd.bdate_range('2021-01-04', periods=120))
    calls = []
    def history(lag, freq, field, stock, df=True):
        calls.append({'lag': lag, 'freq': freq, 'field': field, 'stock': stock, 'df': df})
        return frame.rename(columns={'close': stock}).copy()
    try:
        original_calc(directory, history)(None)
    except TypeError as exc:
        require('method' in str(exc) and 'fillna' in str(exc), 'Unexpected original error')
        failure = {'exception': type(exc).__name__, 'message': str(exc)}
    else:
        raise AssertionError('Expected native pandas compatibility failure')
    prices = frame.resample('D').last().iloc[:, 0]
    current = prices.pct_change(fill_method=None).fillna(0)
    forward = prices.ffill().pct_change(fill_method=None).fillna(0)
    different = current.ne(forward); first = different[different].index[0]
    return {'pandas': pd.__version__, 'numpy': np.__version__, 'original_failure': failure,
            'history_calls_before_failure': calls, 'synthetic_business_rows': len(frame),
            'synthetic_calendar_rows': len(prices), 'changed_return_rows': int(different.sum()),
            'first_example': {'date': str(first.date()), 'current_default_return': float(current.loc[first]),
                              'explicit_forward_filled_return': float(forward.loc[first])},
            'decision': 'pending_D-allweather-fill', 'synthetic_only': True,
            'original_platform_pandas_version_proven': False, 'not_a_backtest': True}


def preflight(root, directory):
    bound = binding(); inputs = candidate_inputs(); dep = inventory(root)
    checkpoint(root, directory, 'Batch89 source stage starts; accepted258 retained; no repaired ES or trades')
    save(directory/'input-binding.json', bound); folder = directory/'sources'; folder.mkdir()
    rows = []
    for name, sha in zip(SOURCES, SOURCE_SHA, strict=True):
        source = Path('repo/量化策略源代码')/name; require(file_sha(source) == sha, 'Source changed')
        text, encoding = read_source(source); tree = ast.parse(text)
        require(hashlib.sha256(ast.dump(tree).encode()).hexdigest() == AST_SHA, 'Duplicate AST changed')
        copied = folder/(strategy_id(name)+'.source'); shutil.copyfile(source, copied)
        rows.append({'source_path': str(source), 'source_copy': str(copied), 'source_sha256': sha,
                     'encoding': encoding, 'lines': len(text.splitlines()), 'ast_sha256': AST_SHA,
                     'definitions': definitions(tree), 'formal_acceptance': False})
    save(directory/'sources-frozen.json', rows); save(directory/'reused-inputs.json', inputs)
    save(directory/'dependency-inventory.json', dep); save(directory/'candidate-coverage.json', candidate_coverage(inputs))
    return {'status': 'ok', 'formal_acceptance': False, 'new_trading_backtests': 0}


def study(root, directory):
    validate_sources(directory)
    require(candidate_inputs() == read(directory/'reused-inputs.json') and
            inventory(root) == read(directory/'dependency-inventory.json'), 'Input inventory changed')
    def forbidden(*a, **kw):
        raise AssertionError('Network forbidden')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden), \
            patch.object(requests.sessions.Session, 'request', forbidden):
        result = compatibility(directory)
        require(candidate_coverage(read(directory/'reused-inputs.json')) == read(directory/'candidate-coverage.json'),
                'Candidate coverage replay differs')
    save(directory/'compatibility-diagnostics.json', {**result, 'socket_HTTP_disabled': True})
    return archive(root, 'Batch89 two same-AST all-weather sources/4candidate histories/compatibility error archived; return repair pending; no trades')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'study'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str))
