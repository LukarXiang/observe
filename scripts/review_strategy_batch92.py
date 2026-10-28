"""Stage the original trend-score notebook on frozen index rows, without trading."""
import argparse
import ast
from collections import defaultdict, namedtuple
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import shutil
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from observe.data.store import Store, fingerprint
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

DIRECTORY = Path('data/staging/strategies-batch92/20261010-trend-definition-stage')
SNAPSHOT = '20261006-145049-e06c'
SOURCE = Path('repo/量化策略源代码/2023年度精选策略/80.趋与势的量化定义.txt')
SOURCE_SHA = '47f4244f0dd45f37cbb6c30c781d47ca7b20b9324ce69587906961a324f8dadd'
PARENT = Path('docs/handoff/2026-10-10-batch91-final-review.json')
FILES = ('scripts/review_strategy_batch92.py', 'tests/unit/test_batch92_stage.py')


def definitions(tree):
    return [{'name': n.name, 'kind': type(n).__name__, 'line': n.lineno, 'end_line': n.end_lineno,
        'ast': ast.dump(n, include_attributes=False)} for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]


def tree(directory=None):
    require(file_sha(SOURCE) == SOURCE_SHA, 'Original source changed')
    if directory is not None: require(file_sha(directory/'original.source') == SOURCE_SHA, 'Frozen source changed')
    return ast.parse(read_source(SOURCE)[0])


def namespace(directory=None):
    original = tree(directory)
    nodes = [n for n in original.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name != 'plot_ochl']
    require([n.name for n in nodes] == ['Normalize_Trend', 'Tren_Score', 'get_act_trend_score',
        'compare_score', 'calc_trend_score', 'get_hold_flag'], 'Original definition scope changed')
    values = dict(pd=pd, np=np, defaultdict=defaultdict, namedtuple=namedtuple, Tuple=Tuple,
        List=List, Callable=Callable, Union=Union, Dict=Dict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<SHA-pinned-original-trend-definitions>', 'exec'), values)
    return values


def parent_binding():
    parent = read(PARENT)
    require(parent['status'] == 'ok' and parent['accepted_checkpoint'] == '20261010-162223-81d4' and
        parent['formal_source_research_acceptance'] and not parent['formal_trading_acceptance'], 'Accepted91 scope changed')
    for prefix in ('receipt', 'checked_state', 'final_binding'):
        require(file_sha(parent[prefix+'_file']) == parent[prefix+'_sha256'], 'Accepted91 binding changed')
    checked = read(parent['checked_state_file'])
    require(all(file_sha(n) == sha for n, sha in checked['implementation_sha256'].items()), 'Accepted91 code changed')
    require(all(file_sha(Path(parent['accepted_directory'])/n) == sha for n, sha in checked['evidence_sha256'].items()),
        'Accepted91 evidence changed')
    return {'parent_file': str(PARENT), 'parent_sha256': file_sha(PARENT), 'snapshot': SNAPSHOT,
        'source_sha256': SOURCE_SHA, 'not_a_backtest': True}


def binding(directory):
    bound = read(directory/'input-binding.json'); expected = parent_binding()
    require(all(bound[n] == v for n, v in expected.items()), 'Input binding changed')
    require(file_sha(bound['snapshot_file']) == bound['snapshot_sha256'], 'Frozen snapshot changed')
    for row in bound['index_partitions']:
        require(file_sha(row['file']) == row['sha256'], 'Frozen index partition changed')
    require(file_sha(directory/'index-input.parquet') == bound['index_copy_sha256'], 'Frozen index operand changed')
    require(definitions(tree(directory)) == read(directory/'source-definition-review.json')['definitions'], 'Original AST evidence changed')
    return bound


def stage(root, directory):
    bound = parent_binding(); original = tree(); text, encoding = read_source(SOURCE)
    store = Store(root); state = store.state(SNAPSHOT)
    require(store.published()['batch_id'] == state['batch_id'] and store.published()['tables'] == state['tables'], 'Publication changed')
    require(state['batch_id'] == '20261006-145049-5ceb', 'Frozen publication changed')
    catalog = read(Path(read(Path(root)/'catalog/strategies/latest.json')['directory'])/'catalog.json')
    row = next(r for r in catalog if r['path'] == SOURCE.relative_to('repo/量化策略源代码').as_posix())
    require(row['review_status'] != '人工审查完成' and row['bytes_sha256'] == SOURCE_SHA, 'Source already reviewed or changed')
    frame = store.load_state(state, 'index_1d', filters=[('index', '=', '000300.SH')])
    frame['date'] = pd.to_datetime(frame.date); frame = frame.sort_values('date').reset_index(drop=True)
    require(len(frame) == 5280 and not frame.date.duplicated().any() and frame['index'].eq('000300.SH').all() and
        frame.date.min() == pd.Timestamp('2005-01-05') and frame.date.max() == pd.Timestamp('2026-09-29') and
        np.isfinite(frame.close).all() and frame.close.gt(0).all(), 'Declared index coverage changed')
    partitions = []
    for item in state['tables']['index_1d'].values():
        path = Path(root)/item['file']; partition = pd.read_parquet(path)
        require(fingerprint(partition) == item['sha'] and len(partition) == item['rows'], 'Original index logical fingerprint differs')
        partitions.append({'file': str(path), 'sha256': file_sha(path), 'dataframe_fingerprint': item['sha'], 'registered_rows': item['rows']})
    checkpoint(root, directory, 'Batch92 original807-line trend notebook and5280 index rows frozen; source research in progress, no trades')
    shutil.copyfile(SOURCE, directory/'original.source')
    frame.to_parquet(directory/'index-input.parquet', index=False)
    snapshot = Path(root)/'snapshots'/f'{SNAPSHOT}.json'
    save(directory/'input-binding.json', {**bound, 'snapshot_file': str(snapshot), 'snapshot_sha256': file_sha(snapshot),
        'index_partitions': partitions, 'index_copy_sha256': file_sha(directory/'index-input.parquet')})
    save(directory/'source-definition-review.json', {'source': str(SOURCE), 'source_sha256': SOURCE_SHA,
        'complete_lines_read': len(text.splitlines()), 'encoding': encoding, 'definitions': definitions(original),
        'full_ast_sha256': hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
        'top_level_executed': False, 'new_trading_backtests': 0,
        'rules': {'MA_rows': 5, 'score_rows': 60, 'quantile_rows': 20, 'lower_quantile': .05, 'upper_quantile': .85,
            'entry': 'previous_score<previous_upper and score>=upper',
            'exit_first': 'previous_score>previous_lower and score<=lower',
            'first_flag': 0, 'absolute_points': 'All global min/max locations plus first/last; no recursive decomposition',
            'paper_return': 'flag * close.pct_change().shift(-1); no costs/orders or executable index'}})
    save(directory/'dependency-inventory.json', {'index_rows': len(frame), 'index_range': [str(frame.date.min().date()), str(frame.date.max().date())],
        'literal_601388_demo': 'not yet inventoried or requested; not silently replaced',
        'missing_plotting_module': 'Hugos_tools.plot_finance', 'full_notebook_compatibility_proven': False,
        'new_requests': 0, 'new_trading_backtests': 0})
    return {'status': 'staged', 'rows': len(frame), 'source_lines': len(text.splitlines()), 'new_trading_backtests': 0}


def sign(value): return int(value > 0)-int(value < 0)


def independent_score(closes):
    # Every 60-row endpoint starts a fresh path: one initial zero, then56 compound steps.
    require(len(closes) == 60 and all(math.isfinite(x) and x > 0 for x in closes), 'Independent score requires60 finite positive rows')
    displacement = [0.]
    for i in range(4, len(closes)):
        direction = sign(closes[i]-closes[i-1])
        ma_direction = sign(closes[i]-math.fsum(closes[i-4:i+1])/5)
        displacement.append(displacement[-1]+(direction+ma_direction)/2)
    low, high = min(displacement), max(displacement)
    selected = [value for i, value in enumerate(displacement) if i in (0, len(displacement)-1) or value in (low, high)]
    return math.fsum((b-a)**2 for a, b in zip(selected, selected[1:]))


def independent_quantile(values, q):
    ordered = sorted(values); index = (len(ordered)-1)*q; lower = math.floor(index)
    if lower == len(ordered)-1: return ordered[lower]
    return ordered[lower]+(ordered[lower+1]-ordered[lower])*(index-lower)


def independent_flags(frame):
    values = [0.]; held = 0
    for i in range(1, len(frame)):
        before, current = frame.iloc[i-1], frame.iloc[i]
        if before.score > before.lower_bound and current.score <= current.lower_bound: held = 0
        elif before.score < before.upper_bound and current.score >= current.upper_bound: held = 1
        values.append(float(held))
    return pd.Series(values, index=frame.index)


def compute(directory):
    binding(directory); ns = namespace(directory); data = pd.read_parquet(directory/'index-input.parquet').set_index('date')
    outputs = []; summaries = []
    for label, close in (('original_period', data.loc['2014-01-01':'2021-10-25', 'close']), ('extended_reference', data.close)):
        score = close.rolling(60).apply(ns['calc_trend_score'], raw=False)
        lower = score.rolling(20).apply(lambda x: x.quantile(.05), raw=False)
        upper = score.rolling(20).apply(lambda x: x.quantile(.85), raw=False)
        flags_input = pd.DataFrame({'score': score, 'lower_bound': lower, 'upper_bound': upper})
        flag = ns['get_hold_flag'](flags_input)
        checked = pd.Series(np.nan, index=close.index)
        for i in range(59, len(close)): checked.iloc[i] = independent_score(close.iloc[i-59:i+1].tolist())
        score_differences = score.ne(checked) & ~(score.isna() & checked.isna())
        quantile_error = {}
        for column, quantile in (('lower_bound', .05), ('upper_bound', .85)):
            expected = pd.Series(np.nan, index=close.index)
            for i in range(78, len(close)): expected.iloc[i] = independent_quantile(score.iloc[i-19:i+1].tolist(), quantile)
            quantile_error[column] = float((flags_input[column]-expected).abs().max())
        checked_flag = independent_flags(flags_input)
        require(flag.equals(checked_flag), 'Independent original signal state differs')
        paper_next = close.pct_change().shift(-1)
        table = flags_input.assign(flag=flag, independent_score=checked, paper_next_close_return=paper_next,
            paper_flag_times_return=flag*paper_next, range=label).reset_index()
        outputs.append(table)
        summaries.append({'range': label, 'rows': len(close), 'first': str(close.index.min().date()), 'last': str(close.index.max().date()),
            'finite_scores': int(score.notna().sum()), 'finite_bounds': int(lower.notna().sum()),
            'entry_transitions': int(flag.diff().eq(1).sum()), 'exit_transitions': int(flag.diff().eq(-1).sum()),
            'held_observation_rows': int(flag.eq(1).sum()), 'independent_score_differences': int(score_differences.sum()),
            'independent_score_difference_dates': [str(d.date()) for d in close.index[score_differences]],
            'max_independent_quantile_abs_error': quantile_error, 'independent_flags_differences': 0,
            'last_next_return_missing': bool(pd.isna(paper_next.iloc[-1])), 'not_a_backtest': True})
    return pd.concat(outputs, ignore_index=True), {'status': 'component_research', 'ranges': summaries,
        'overlapping_ranges': True, 'original_60_row_branch': 'absolute only; relative/mixed demo failures are not repaired',
        'signal_rows_are_index_paper_states': True, 'new_trading_backtests': 0, 'net_asset_value_computed': False,
        'platform_equivalence_proven': False, 'source_body_changed': False}


def component(root, directory):
    frame, result = network_disabled(lambda: compute(directory))
    require(not (directory/'trend-values.parquet').exists(), 'Component values already archived')
    frame.to_parquet(directory/'trend-values.parquet', index=False)
    save(directory/'component-research.json', result)
    progress = archive(root, 'Batch92 original trend score/index paper-state component archived; pending checks and source acceptance; no trading backtest')
    save(directory/'component-checkpoint.json', progress)
    return result


def offline(_root, directory):
    before = read(directory/'component-research.json')
    frame, result = network_disabled(lambda: compute(directory))
    require(result == before, 'Offline component summary differs')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(directory/'trend-values.parquet'), check_exact=True)
    proof = {'status': 'match', 'differences': 0, 'rows': len(frame), 'socket_HTTP_disabled': True,
        'not_a_backtest': True, 'formal_acceptance': False, 'new_trading_backtests': 0}
    save(directory/'offline-verification.json', proof); return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'component', 'offline'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    with redirect_stdout(StringIO()): result = globals()[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False))
