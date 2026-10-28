"""Accept two source reviews against actual arithmetic/recovery artifacts, not trades."""
import argparse
import json
import math
from pathlib import Path
import pprint
import shutil
import sys

import pandas as pd
import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts.archive_strategy_progress import archive
from scripts import diagnose_strategy_batch99_timing as timing
from scripts import research_strategy_batch99 as stage
from scripts import recover_strategy_batch99_namespace as recovery
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch99/20261011-formal-source-acceptance')
PHASE = Path('docs/handoff/2026-10-11-batch99-arithmetic-stage.json')
PHASE_SHA = '7323d0565a88e4a199e91710c2f07c0c932533fcb3daf0ee3a504ff0ffe95739'
SPECS = ('docs/handoff/batch99-y2021-064-source-review.yaml',
         'docs/handoff/batch99-y2022-018-source-review.yaml')
GUARD = Path('data/staging/strategies-batch99/20261011-formal-validation-v2/checked.json')
RECEIPT = Path('docs/handoff/2026-10-11-batch99-final-review.json')
REPORT = Path('docs/tasks/2026-10-11-第九十九批中国VIX与期权突破-正式研究验收.md')
EXPECTED_ERRORS = {'less_than_two_expiries': 'UnboundLocalError', 'missing_put_field': 'NameError',
    'single_strike_deltaK': 'IndexError', 'equal_maturity_division_by_zero': 'ZeroDivisionError',
    'zero_forward_epsilon_nonfinite': 'ZeroDivisionError', 'shibor_2001days_unbound_local': 'UnboundLocalError',
    'shibor_string_date_attribute_error': 'AttributeError', 'GetCBOE_VIX_pandas3_pad_compatibility': 'TypeError',
    'preopen_short_TradingCode': 'IndexError', 'preopen_first_expiry_adjusted_only_no_next_fallback': 'KeyError',
    'Indicators_sqrt_not_imported': 'NameError', 'paper_zero_premium': 'OverflowError',
    'paper_missing_entry': 'ValueError'}


def reviews():
    result = {}
    for i, name in enumerate(SPECS):
        row = yaml.safe_load(Path(name).read_text())
        require(row['source'] == stage.SOURCE_PATHS[i] and row['source_sha256'] == stage.SOURCE_SHA[i] and
                row['formal_acceptance'] is False and row['original_strategy_complete'] is False and
                row['economic_variant_approved'] is False, 'Frozen phase rules identity differs')
        result[row['source']] = {'review_status': '人工审查完成', 'scope': row['review_scope'],
            'status': '待数据' if i == 0 else '非交易研究脚本', 'implementations': [],
            'original_strategy_complete': False, 'fidelity': row['allowed_evidence'], 'rules': row['rules'],
            'gaps': row['data_gaps']+row['defects'], 'next_step': row['next']}
    return result


def candidate_checks(old, new, selected):
    require(len(old) == len(new) == 695 and len(selected) == 2, 'Catalog scope differs')
    previous = {r['path']: r for r in old}
    require(len(previous) == len({r['path'] for r in new}) == 695 and
            {r['path'] for r in new} == set(previous), 'Catalog identities differ')
    require({r['path'] for r in new if r != previous[r['path']]} == set(selected), 'Unrelated/missing catalog change')
    for row in new:
        if row['path'] not in selected: continue
        before = previous[row['path']]; allowed = selected[row['path']]
        require(before['review_status'] != '人工审查完成' and
            {k:v for k,v in row.items() if k not in allowed} ==
            {k:v for k,v in before.items() if k not in allowed}, 'Source static identity changed')
        require(all(row[k] == value for k,value in allowed.items()), 'Reviewed rule differs')
    require(sum(r['review_status'] == '人工审查完成' for r in new) == 284, 'Manual count differs')
    return {'sources': 695, 'reviewed': 284, 'remaining': 411, 'changed': sorted(selected)}


def catalog_program(before, selected):
    marker = b'REVIEWS = {\n'; require(before.count(marker) == 1, 'Catalog insertion marker differs')
    return before.replace(marker, marker+(pprint.pformat(selected, sort_dicts=False, width=100)[1:-1]+',\n').encode(), 1)


def diagnostic_checks(doc):
    rows = {r['case']: r for r in doc['cases']}
    require(len(rows) == len(doc['cases']) == doc['case_count'] == 44 and
            doc['new_trading_backtests'] == 0 and doc['actual_json_roundtrip'] is True and
            doc['historical_option_chain_used'] is False, 'Diagnostic scope differs')
    require({k:v['error'] for k,v in rows.items() if v['error']} == EXPECTED_ERRORS, 'Original exceptions differ')
    expected = {'five_day_expiry_included': {'jy': 5/365, 'cjy': 40/365},
        'K0_is_min_diff_not_largest_strike_below_F': 2., 'EMA_equal_buys': True,
        'EMA_rising_sells': False, 'EMA_falling_buys': True, 'EMA_short_window_false': False,
        'call_put_order': [['low', 'mid', 'high'], ['high', 'mid', 'low']],
        'edge_strike_only_two': ['low', 'mid'], 'preopen_empty_None_strings': ['None', 'None'],
        'negative_variance_returns_nan': {'nonfinite': 'nan'},
        'shibor_first_page_is_one_day': [{'start_date': '20050101', 'end_date': '20050101'}]}
    require(all(rows[n]['value'] == v for n,v in expected.items()), 'Original arithmetic/state boundary differs')
    for name in ('scalar_vix_skew_comparison',):
        require(isinstance(rows[name]['value'], list) and len(rows[name]['value']) == 2 and
                all(v['sigma_error'] < 1e-12 and v['skew_error'] < 1e-10 for v in rows[name]['value']),
                'Independent VIX/SKEW calculation differs')
    for name in ('combined_vix_skew_original_weights', 'both_expiries_after30_weight_unclamped'):
        value = rows[name]['value']
        require(value['VIX_error'] < 1e-10 and value['SKEW_error'] < 1e-10, 'Combined moments differ')
    require(rows['both_expiries_after30_weight_unclamped']['value']['weight'] == 2., 'Original unclamped weight differs')
    paper = [r for r in doc['cases'] if r['case'].startswith('paper_')]
    require(len(paper) == 14 and all(r['scope'] == 'paper_callback_not_Book' for r in paper), 'Paper scope differs')
    for direction in ('up', 'down'):
        for first, second in (('Buy', 'Sell'), ('Sell', 'Buy'), ('Buy', 'Buy'), ('Sell', 'Sell')):
            value = rows['paper_'+direction+'_'+first+'_'+second]['value']
            buy = (first if direction == 'up' else second) == 'Buy'
            pin, pout = ((.02, .03) if direction == 'up' and buy else (.03, .01) if direction == 'up'
                         else (.03, .01) if buy else (.02, .03))
            qty = int(1000/pin/10000); profit = ((pout-pin) if buy else (pin-pout))-.001
            require(isinstance(value, list) and len(value) == 2 and len(value[0]) == 1 and
                    len(value[1]) == 2 and abs(value[1][-1]-(1+qty*profit)) < 1e-12, 'Paper first-breakout result differs')
    require(rows['paper_state_resets_next_day']['value'][1] == [1., 1.045, .99], 'Daily state reset differs')
    return {'cases': 44, 'preserved_original_exceptions': 13, 'synthetic_only': True, 'fills': 0}


def range_checks(frame, operand, summary):
    require(len(operand) == 5252 and len(frame) == summary['rows'] == 5249 and
            frame.date.tolist() == operand.date.iloc[3:].tolist() and
            frame.last_previous_date.tolist() == operand.date.iloc[2:-1].tolist() and
            summary['first'] == '2005-02-28' and summary['last'] == '2026-09-29' and
            summary['new_trading_backtests'] == 0 and summary['source_candidate_strict_usable'] is False,
            'Band dates/scope differ')
    for i, row in enumerate(frame.itertuples(index=False), start=3):
        window = operand.iloc[i-3:i]
        width = max(max(map(float, window.high))-min(map(float, window.close)),
                    max(map(float, window.close))-min(map(float, window.low)))
        opening = float(operand.open.iloc[i])
        require(row.range == width and row.open == opening and
                row.scalar_up == round(opening+.3*width, 3) and row.scalar_down == round(opening-.3*width, 3),
                'Independent scalar reference differs')
        require(abs(row.original_up-row.scalar_up) <= .001000000001 and
                abs(row.original_down-row.scalar_down) <= .001000000001,
                'Original bands exceed disclosed rounding edges')
    observed = {k: int((frame.original_up.ne(frame[k+'_up']) | frame.original_down.ne(frame[k+'_down'])).sum())
                for k in ('scalar', 'decimal')}
    require(observed == summary['rounding_difference_days'] == {'scalar': 319, 'decimal': 151}, 'Rounding disclosure differs')
    example = frame.loc[frame.date.eq('2018-07-20')].iloc[0]
    require(example.original_up == 2.487 and example.original_down == 2.461 and
            example.open == 2.474 and math.isclose(example.range, .042), 'Original dated example differs')
    return {'rows': 5249, 'calendar_price_rows': 5252, 'rounding_edges': observed, 'signal_or_trade': False}


def timing_checks(doc):
    require(doc['case_count'] == len(doc['cases']) == 8 and doc['source_sha256'] == stage.SOURCE_SHA[0] and
            doc['new_trading_backtests'] == 0 and doc['real_option_history_used'] is False, 'Timing scope differs')
    rows = {r['case']: r['calls'] for r in doc['cases']}
    require(len(rows) == 8, 'Duplicate callback case')
    for key in ('flat_buy_empty', 'flat_buy_held', 'rising_sell_empty', 'rising_sell_held'):
        require(rows[key][:2] == [['get_trade_days', {'end_date': '2018-07-20', 'count': 15}],
            ['CH_VIX', ['510050.XSHG', '2018-07-02', '2018-07-20']]], 'Decision cutoff differs')
    require(rows['flat_buy_empty'][-1] == ['order_target_value', ['510050.XSHG', 123456.]] and
            len(rows['flat_buy_held']) == len(rows['rising_sell_empty']) == 3 and
            rows['rising_sell_held'][-1] == ['order_target', ['510050.XSHG', 0]], 'Holding/cash intent differs')
    for date,buy,sell in (('2008-12-31', .003, .004), ('2010-01-04', .002, .003),
                          ('2012-01-04', .001, .002), ('2015-02-09', .0003, .0013)):
        require(rows['fees_'+date] == [['slippage', 0], ['commission',
            {'buy_cost': buy, 'sell_cost': sell, 'min_cost': 5}]], 'Dated source commission differs')
    return {'cases': 8, 'intents_not_fills': True, 'platform_equivalence': False}


def components():
    require(file_sha(PHASE) == PHASE_SHA, 'Phase receipt changed')
    phase = read(PHASE)
    for name, sha in phase['evidence_sha256'].items(): require(file_sha(name) == sha, 'Phase evidence changed: '+name)
    for folder in (stage.DIRECTORY, recovery.DIRECTORY):
        prepared = read(folder/'prepared.json')
        for name, sha in prepared['implementation_sha256'].items():
            require(file_sha(folder/'implementation'/name) == sha, 'Frozen implementation changed')
            if name != 'src/observe/strategy_catalog.py': require(file_sha(name) == sha, 'Runtime helper changed')
        require(file_sha(folder/'ETF-input.parquet') == prepared['operand_sha256'], 'Operand bytes differ')
    for n, path in enumerate(stage.SOURCE_PATHS):
        require(file_sha(Path('repo/量化策略源代码')/path) ==
                file_sha(stage.DIRECTORY/'sources'/f'{n}.source') == stage.SOURCE_SHA[n], 'Source identity differs')
    require(stage.inventory() == read(stage.DIRECTORY/'dependencies.json'), 'Active input inventory changed')
    off = read(recovery.DIRECTORY/'offline-verification.json')
    require(off['status'] == 'match' and off['differences'] == 0 and off['fresh_process'] and off['network_blocked'] and
            set(off['output_sha256']) == set(stage.OUTPUTS), 'Recovery offline proof differs')
    for mode in ('study', 'replay'):
        command = read(recovery.DIRECTORY/(mode+'-command')/'command.json')
        require(command['returncode'] == 0 and command['command'] == [sys.executable, '-B', '-m',
            'scripts.recover_strategy_batch99_namespace', 'worker', '--mode', mode], 'Actual worker command differs')
        for stream in ('stdout', 'stderr'):
            require(file_sha(recovery.DIRECTORY/(mode+'-command')/(stream+'.log')) == command[stream+'_sha256'],
                    'Worker stream differs')
        for name, sha in off['output_sha256'].items():
            require(file_sha(recovery.DIRECTORY/mode/name) == sha, 'Worker outputs differ')
    frame = pd.read_parquet(recovery.DIRECTORY/'study/ETF-range-components.parquet')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(recovery.DIRECTORY/'replay/ETF-range-components.parquet'), check_exact=True)
    ranges = range_checks(frame, pd.read_parquet(recovery.DIRECTORY/'ETF-input.parquet'), read(recovery.DIRECTORY/'study/summary.json'))
    diagnostics = diagnostic_checks(read(recovery.DIRECTORY/'study/diagnostics.json'))
    proof = read(timing.DIRECTORY/'offline.json')
    require(proof['status'] == 'match' and proof['differences'] == 0 and proof['fresh_process'] and proof['network_blocked'] and
            file_sha(timing.DIRECTORY/'study.json') == file_sha(timing.DIRECTORY/'replay.json') == proof['sha256'],
            'Timing fresh replay differs')
    for mode in ('study', 'replay'):
        command = read(timing.DIRECTORY/(mode+'-command.json'))
        require(command['returncode'] == 0 and command['command'] == [sys.executable, '-B', '-m',
                'scripts.diagnose_strategy_batch99_timing', 'worker', '--mode', mode], 'Timing command differs')
        for stream in ('stdout', 'stderr'):
            require(file_sha(timing.DIRECTORY/(mode+'.'+stream+'.log')) == command[stream+'_sha256'],
                    'Timing command stream differs')
    for name,sha in read(timing.DIRECTORY/'prepared.json')['implementation_sha256'].items():
        require(file_sha(name) == sha, 'Timing frozen implementation changed')
    callbacks = timing_checks(read(timing.DIRECTORY/'study.json'))
    guard = read(GUARD)
    require(guard['status'] == 'ok' and guard['local_passed'] == 66 and guard['isolated_passed'] == 43 and
            guard['skips'] == 0 and all(c['returncode'] == 0 for c in guard['commands']), 'Formal validation scope incomplete')
    for name, sha in {**guard['implementation_sha256'], **guard['evidence_sha256']}.items():
        if name != 'src/observe/strategy_catalog.py': require(file_sha(name) == sha, 'Formal guard changed: '+name)
    isolation = read(GUARD.parent/'isolated.json'); root = Path(isolation['cwd'])
    require(isolation['tests_passed'] == 43 and isolation['skips'] == 0 and len(isolation['actual_imports']) == 7 and
            len(isolation['copied_sha256']) == guard['copied_files'] and
            all(Path(p).is_relative_to(root) for p in isolation['actual_imports'].values()), 'Isolated actual scope differs')
    for name,sha in isolation['copied_sha256'].items(): require(file_sha(root/name) == sha, 'Isolated checked file changed')
    fresh = read(GUARD.parent/'fresh-replay.json')
    require(fresh['status'] == 'match' and fresh['differences'] == 0 and fresh['fresh_process'] and
            fresh['network_blocked'] and fresh['command']['returncode'] == 0 and
            fresh['output_sha256'] == off['output_sha256'], 'Further fresh recovery replay differs')
    for name,sha in fresh['output_sha256'].items():
        require(file_sha(GUARD.parent/'fresh-replay/replay'/name) == sha, 'Fresh replay artifact changed')
    require(read(stage.DIRECTORY/'protection.json')['old_controls_unchanged'] == 6614,
            'Old protection proof differs')
    return {'ranges': ranges, 'diagnostics': diagnostics, 'callbacks': callbacks, 'offline': 'match/0',
            'local_tests': guard['local_passed'], 'isolated_tests': guard['isolated_passed'],
            'new_trading_backtests': 0, 'original_strategy_complete': False}


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal output exists')
    result = components(); selected = reviews(); active = read('data/catalog/strategies/latest.json')
    require(active['catalog_id'] == read(stage.PARENT)['progress']['catalog_id'], 'Parent catalog differs')
    old = read(Path(active['directory'])/'catalog.json')
    before = (stage.DIRECTORY/'implementation/src/observe/strategy_catalog.py').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == before, 'Catalog changed before preparation')
    DIRECTORY.mkdir(parents=True); save(DIRECTORY/'source-reviews.json', selected)
    shutil.copyfile(stage.DIRECTORY/'baseline.json', DIRECTORY/'baseline.json')
    REVIEWS.update(selected); digests = []
    for tag in ('candidate', 'candidate-replay'):
        root = DIRECTORY/tag; target = root/'catalog/strategies/implementation-evidence.json'
        target.parent.mkdir(parents=True); shutil.copyfile(stage.DIRECTORY/'implementation-evidence.before.json', target)
        candidate = catalog_strategies(root, 'repo/量化策略源代码'); output = Path(candidate['output'])
        candidate_checks(old, read(output/'catalog.json'), selected)
        digests.append({n: file_sha(output/n) for n in ('catalog.json', 'summary.json', 'catalog.parquet')})
        if tag == 'candidate': proposed = candidate
    require(digests[0] == digests[1], 'Candidate byte replay differs')
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(catalog_program(before, selected))
    paths = [*SPECS, str(PHASE), str(GUARD), str(timing.DIRECTORY/'offline.json'),
        str(timing.DIRECTORY/'study.json'), str(recovery.DIRECTORY/'offline-verification.json'),
        str(recovery.DIRECTORY/'prepared.json'), str(DIRECTORY/'source-reviews.json'),
        'scripts/accept_strategy_batch99.py', 'tests/unit/test_batch99_acceptance.py']
    save(DIRECTORY/'prepared.json', {'components': result, 'candidate': proposed, 'catalog_bytes': digests[0],
        'previous_catalog': active, 'evidence_sha256': {n: file_sha(n) for n in paths},
        'proposed_code_sha256': file_sha(DIRECTORY/'strategy_catalog.proposed.py')})
    return {'status': 'candidate_verified', 'not_published': True, 'catalog_id': proposed['catalog_id']}


def publish():
    require(not RECEIPT.exists(), 'Immutable formal receipt exists')
    doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name, sha in doc['evidence_sha256'].items(): require(file_sha(name) == sha, 'Prepared evidence changed')
    require(components() == doc['components'] and REPORT.is_file(), 'Formal evidence/report differs')
    before = (stage.DIRECTORY/'implementation/src/observe/strategy_catalog.py').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == catalog_program(before, selected) and
            file_sha('src/observe/strategy_catalog.py') == doc['proposed_code_sha256'] and
            all(REVIEWS.get(n) == value for n,value in selected.items()), 'Proposed runtime catalog differs')
    candidate = Path(doc['candidate']['output'])
    candidate_checks(read(Path(doc['previous_catalog']['directory'])/'catalog.json'), read(candidate/'catalog.json'), selected)
    for name, sha in doc['catalog_bytes'].items(): require(file_sha(candidate/name) == sha, 'Candidate bytes differ')
    with operation_lock('data', DATA_WRITER):
        latest = Path('data/catalog/strategies/latest.json'); progress_latest = Path('data/catalog/strategies/progress/latest.json')
        require(read(latest) == doc['previous_catalog'], 'Active catalog pointer changed')
        old_latest, old_progress = latest.read_bytes(), progress_latest.read_bytes()
        proof = read(DIRECTORY/'protection.json')
        require(proof['status'] == 'ok' and proof['old_controls_unchanged'] == 6614 and
                proof['old_partition_metadata_unchanged'] == 398, 'Fresh formal protection missing')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']; dest.mkdir(exist_ok=False)
        for name, sha in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/name, dest/name); require(file_sha(dest/name) == sha, 'Published bytes differ')
        try:
            _atomic_json(latest, {'catalog_id': dest.name, 'directory': str(dest), 'sha256': doc['catalog_bytes']['catalog.json']})
            progress = archive('data', 'Batch99 C-VIX64+option18 source research accepted;5249 candidate bands/44+8 original diagnostics/recovery offline match0;option trades blocked')
            require((progress['manually_reviewed'],progress['locally_backtested_sources'],progress['locally_reproduced_sources'],
                    progress['local_strategy_runs']) == (284,12,12,87), 'Formal source/trading counts differ')
            bound = {**doc['evidence_sha256'], str(DIRECTORY/'prepared.json'): file_sha(DIRECTORY/'prepared.json'),
                str(REPORT): file_sha(REPORT), 'src/observe/strategy_catalog.py': file_sha('src/observe/strategy_catalog.py'),
                str(DIRECTORY/'protection.json'): file_sha(DIRECTORY/'protection.json'),
                str(dest/'catalog.json'): file_sha(dest/'catalog.json'),
                str(Path(progress['output'])/'summary.json'): file_sha(Path(progress['output'])/'summary.json')}
            require(read(latest)['catalog_id'] == progress['catalog_id'] == dest.name and
                    read(progress_latest)['checkpoint'] == progress['checkpoint'] and
                    all(file_sha(n) == s for n,s in bound.items()), 'Final binding/navigation readback failed')
            save(RECEIPT, {'status': 'ok', 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
                'library_goal_complete': False, 'original_strategy_complete': False, 'progress': progress,
                'components': doc['components'], 'source_sha256': list(stage.SOURCE_SHA), 'evidence_sha256': bound,
                'protection': proof, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
                'phase_specs_remain_frozen': True, 'review_method': 'Full source rules, independent scalar arithmetic, actual JSON, fresh blocked-network workers and local/isolated checks; no independent agent review'})
        except Exception:
            latest.write_bytes(old_latest); progress_latest.write_bytes(old_progress); raise
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 284, 'remaining': 411}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'protect', 'publish')); args = parser.parse_args()
    result = protect('data', DIRECTORY) if args.action == 'protect' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
