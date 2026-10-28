"""Accept two source reviews with actual longest feature evidence, never model trades."""
import argparse
import json
import math
from pathlib import Path
import pprint
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store, _atomic_json
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts.archive_strategy_progress import archive
from scripts import research_strategy_batch100 as study
from scripts import recover_strategy_batch100 as recovery
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch100/20261011-formal-source-acceptance')
RECEIPT = Path('docs/handoff/2026-10-11-batch100-final-review.json')
REPORT = Path('docs/tasks/2026-10-11-第一百批行业轮动与50ETF-Xgboost-正式研究验收.md')
SPECS = ('docs/handoff/batch100-y2022-032-source-review.yaml',
         'docs/handoff/batch100-y2022-026-source-review.yaml')
GUARD = Path('data/staging/strategies-batch100/20261011-formal-validation/checked.json')
PHASES = ('docs/handoff/2026-10-11-batch100-source-stage.json',
          'docs/handoff/2026-10-11-batch100-supplement-stage.json',
          'docs/handoff/2026-10-11-batch100-recovery-stage.json')
ERRORS = {'empty_minute_price': 'IndexError', 'rolling_rank_current_pandas_fails': 'TypeError',
          'xgb_empty_ensemble_error': 'IndexError',
          'xgb_missing_col_50etf_20191109.dat': 'FileNotFoundError',
          'xgb_missing_model_50etf_20191109.dat': 'FileNotFoundError'}


def reviews():
    selected = {}
    for n, name in enumerate(SPECS):
        doc = yaml.safe_load(Path(name).read_text())
        require(doc['source'] == study.SOURCE_PATHS[n] and doc['source_sha256'] == study.SOURCE_SHA[n] and
                doc['formal_acceptance'] is False and doc['original_strategy_complete'] is False and
                doc['economic_variant_approved'] is False, 'Frozen source review identity differs')
        selected[doc['source']] = {'review_status': '人工审查完成', 'status': '待数据' if n == 0 else '暂不可复现',
            'scope': doc['review_scope'], 'rules': doc['rules'], 'fidelity': doc['allowed_evidence'],
            'gaps': doc['data_gaps']+doc['defects'], 'next_step': doc['next'],
            'implementations': [], 'original_strategy_complete': False}
    return selected


def candidate_checks(old, new, selected):
    require(len(old) == len(new) == 695 and len(selected) == 2, 'Catalog scope differs')
    previous = {r['path']: r for r in old}
    require(len(previous) == len({r['path'] for r in new}) == 695 and
            set(previous) == {r['path'] for r in new}, 'Catalog identities differ')
    require({r['path'] for r in new if r != previous[r['path']]} == set(selected), 'Unrelated/missing source change')
    for row in new:
        if row['path'] not in selected: continue
        before, allowed = previous[row['path']], selected[row['path']]
        require(before['review_status'] != '人工审查完成' and
                {k: v for k, v in row.items() if k not in allowed} ==
                {k: v for k, v in before.items() if k not in allowed}, 'Static source identity changed')
        require(all(row[k] == v for k, v in allowed.items()), 'Literal reviewed rule differs')
    require(sum(r['review_status'] == '人工审查完成' for r in new) == 286, 'Manual count differs')
    return {'sources': 695, 'reviewed': 286, 'remaining': 409, 'changed': sorted(selected)}


def catalog_program(before, selected):
    marker = b'REVIEWS = {\n'; require(before.count(marker) == 1, 'Catalog insertion marker differs')
    return before.replace(marker, marker+(pprint.pformat(selected, sort_dicts=False, width=100)[1:-1]+',\n').encode(), 1)


def diagnostic_checks(doc):
    rows = {r['case']: r for r in doc['cases']}
    require(len(rows) == len(doc['cases']) == doc['case_count'] == 32 and doc['synthetic_only'] is True and
            doc['model_predictions_are_stubs'] is True and doc['real_pickle_executed'] is False and
            doc['actual_json_roundtrip'] is True and doc['new_trading_backtests'] == 0, 'Diagnostic scope differs')
    require({k: v['error'] for k, v in rows.items() if v['error']} == ERRORS, 'Original exception set differs')
    def value(name): return rows[name]['value']
    mapping = value('sector_identity_mapping')
    require(mapping['stocks'] == ['000932.XSHG', '000913.XSHG', '399967.XSHE', '000993.XSHG',
                '000819.XSHG', '399975.XSHE', '399971.XSHE'] and
            mapping['ETFs'] == {s: s for s in mapping['stocks']} and
            (mapping['lag'], mapping['bars'], mapping['time'], mapping['money']) == (10, 20, [14, 15], '511880.XSHG'),
            'Original pool/carrier/minute parameters differ')
    require(value('risk_off_two_positions') == [['target', 'old1', 0], ['unused_price', '511880.XSHG'],
            ['target', 'old1', 0], ['buy', '511880.XSHG', 1234.]] and
            value('risk_off_immediate_double_sell')[-1] == ['buy', '511880.XSHG', 1434.], 'Partial sell/cash intent differs')
    require(value('risk_off_money_held') == value('wrong_minute_no_action') == [] and
            value('risk_on_same_held') == [['unused_price', '511880.XSHG']], 'Original holding/time branch differs')
    for key, expected in (('stable_tie_first_sector_and_minute_counts', '000932.XSHG'),
            ('strict_mean_equality_risk_off', 'sell_the_stocks'), ('all_nan_risk_off', 'sell_the_stocks')):
        result = value(key); calls = result['calls']
        require(result['signal'] == expected and len(calls) == 9 and
            calls[:7] == [[s, 2400, {'unit': '1m', 'fields': 'close', 'skip_paused': True}] for s in mapping['stocks']] and
            calls[-2:] == [['000932.XSHG', count, {'unit': '1m', 'fields': 'close', 'skip_paused': True}]
                for count in (2400, 4800)], 'Minute cutoff/ranking/tie branch differs')
    for date, buy, sell in (('2009-01-01', .003, .004), ('2009-01-02', .002, .003),
            ('2011-01-01', .002, .003), ('2011-01-02', .001, .002),
            ('2013-01-01', .001, .002), ('2013-01-02', .0003, .0013)):
        require(value('fees_'+date) == [['slippage', 0], ['commission',
                {'buy_cost': buy, 'sell_cost': sell, 'min_cost': 5}]], 'Strict fee-date boundary differs')
    matrix = ['matrix', ['b', 'a'], [[3., 1.]]]
    for key in ('xgb_upper_equal', 'xgb_lower_equal', 'xgb_ensemble_mean_hold', 'xgb_nan_hold'):
        require(value(key) == [matrix], 'Strict threshold/ensemble/column-order branch differs')
    require(value('xgb_buy') == [matrix, ['buy', '510050.XSHG', 1234.]] and
            value('xgb_sell') == [matrix, ['target', '510050.XSHG', 0]] and
            value('rank_returns_series') == [2.5] and value('rankeq10_counts_exact_tens') == 2,
            'Original order/rank values or JSON list shapes differ')
    require(value('five_separate_daily_history_calls') == {'calls': [['510050.XSHG', 400, '1d', f]
            for f in ('close', 'high', 'low', 'open', 'volume')], 'return_order_lengths': [400]*5},
            'Daily feature request contract differs')
    require(value('xgb_daily_0950_and_stock_cost') == {'etf': '510050.XSHG', 'calls': [
        ['schedule', 'trade', ['9:50'], {'reference_security': '000300.XSHG'}],
        ['cost', [{'close_tax': .001, 'open_commission': .00025, 'close_commission': .00025,
            'min_commission': 5}], {'type': 'stock'}]]}, 'Original schedule/stock fee declaration differs')
    return {'cases': 32, 'original_exceptions': 5, 'synthetic_only': True, 'real_model_predictions': 0}


def feature_checks(frame, operand, summary, scalar_dates=None):
    require(len(operand) == 5252 and len(frame) == summary['windows'] == 4852 and
            summary['feature_columns'] == len(frame.columns)-3 == 1059 and
            not frame.columns.duplicated().any(), 'Longest feature shape differs')
    require(frame.decision_date.tolist() == operand.date.iloc[400:].tolist() and
            frame.first_input_date.tolist() == operand.date.iloc[:-400].tolist() and
            frame.last_input_date.tolist() == operand.date.iloc[399:-1].tolist() and
            summary['first_decision'] == '2006-10-17' and summary['last_decision'] == '2026-09-29',
            'Feature history/decision cutoff differs')
    require(summary['year_counts'] == frame.groupby(frame.decision_date.str[:4]).size().to_dict() and
            summary['window_rows'] == 400 and summary['candidate_rows'] == 5252 and
            summary['strict_usable'] is False and summary['new_trading_backtests'] == summary['new_strategy_parent_runs'] == 0,
            'Feature year coverage/trading scope differs')
    require(summary['independent_scalar_cells'] == 4852*1059 and
            math.isfinite(summary['max_scalar_relative_error']) and summary['max_scalar_relative_error'] < 2e-8,
            'Producer column comparisons incomplete')
    errors = summary['original_full_pipeline_failures']
    require(len(errors) == 2 and [r['decision_date'] for r in errors] == ['2006-10-17', '2026-09-29'] and
            all(r['error'] == 'TypeError' and 'Series' in r['message'] for r in errors), 'Full original failure differs')
    for name in ('close', 'open', 'high', 'low', 'volume'):
        require(np.array_equal(frame[name].to_numpy(), operand[name].iloc[399:-1].to_numpy()),
                'Saved feature/actual input mismatch: '+name)
    for n in (5, 10, 20):
        require(np.array_equal(frame['long_MA'+str(n)+'_flag'].to_numpy(),
            frame.close.gt(frame['MA'+str(n)]).astype(float).to_numpy()), 'Saved strict MA flag differs')
    for right in (10, 20):
        require(np.array_equal(frame['long_MA5_MA'+str(right)].to_numpy(),
            frame.MA5.gt(frame['MA'+str(right)]).astype(float).to_numpy()), 'Saved strict MA pair differs')
    require('alpha005' not in frame and 'alpha006' not in frame and 'alpha004' in frame and
            'aphla001' in frame and 'alpha001' not in frame, 'Original prefix boundary/name differs')
    # Producer audited every derived tail. This separate readback uses one first window per year and the final one.
    dates = scalar_dates if scalar_dates is not None else sorted(set(frame.groupby(frame.decision_date.str[:4]).decision_date.first()) | {frame.decision_date.iloc[-1]})
    positions = {v: i for i, v in enumerate(frame.decision_date)}; ns = study.original(1)
    checked = 0; max_error = 0.
    for date in dates:
        require(date in positions, 'Readback sample outside result range'); j = positions[date]
        row = frame.iloc[j].drop(labels=['decision_date', 'first_input_date', 'last_input_date'])
        cells, error = study.scalar_checks(ns, operand.iloc[j:j+400].set_index('date'), row)
        checked += cells; max_error = max(max_error, error)
    return {'windows': 4852, 'first': summary['first_decision'], 'last': summary['last_decision'],
        'features': 1059, 'producer_column_comparisons': 4852*1059,
        'producer_independent_derived_comparisons': 4852*1034,
        'base25_comparisons_reuse_original_function_and_library': True,
        'separate_readback_windows': len(dates), 'separate_column_comparisons': checked,
        'readback_max_relative_error': max_error, 'model_input_or_trade': False}


def components():
    study.validate(); recovery_doc = recovery.validate()
    require(Store('data').published()['batch_id'] == read(study.DIRECTORY/'dependencies.json')['published_batch'],
            'Actual data-root publication changed')
    for name in PHASES:
        phase = read(name)
        require(phase['formal_source_research_acceptance'] is False, 'Phase improperly promoted')
        for path, digest in phase['evidence_sha256'].items():
            if path == 'src/observe/strategy_catalog.py': continue
            require(file_sha(path) == digest, 'Phase evidence changed: '+path)
    proof = read(recovery.DIRECTORY/'offline-verification.json')
    require(proof['status'] == 'match' and proof['differences'] == 0 and proof['fresh_process_per_chunk'] is True and
            proof['network_blocked'] is True and set(proof['output_sha256']) == set(study.OUTPUTS) and
            proof['windows_per_mode'] == 4852 and proof['chunks_per_mode'] == 49 and
            proof['recovery_prepared_sha256'] == file_sha(recovery.DIRECTORY/'prepared.json'), 'Fresh longest replay incomplete')
    for mode in ('study', 'replay'):
        assembly = read(recovery.DIRECTORY/mode/'assembled/assembly.json')
        require(assembly['windows'] == 4852 and assembly['chunks'] == 49 and
                assembly['output_sha256'] == proof['output_sha256'], 'Assembly scope differs')
        for a, b in recovery_doc['ranges']:
            folder = recovery.existing(mode, a, b, recovery_doc['operand_sha256'])
            require(folder is not None, 'Missing fresh chunk')
            require(assembly['chunk_receipt_sha256'].get(str(folder/'completed.json')) == file_sha(folder/'completed.json'),
                    'Assembly chunk identity differs')
            command = read(folder/'command.json')
            require(command['command'] == [sys.executable, '-B', '-m', 'scripts.recover_strategy_batch100',
                'worker', '--mode', mode, '--start', str(a), '--end', str(b), '--folder', str(folder)],
                'Observed fresh chunk command differs')
        for name, digest in proof['output_sha256'].items():
            require(file_sha(recovery.DIRECTORY/mode/'assembled'/name) == digest, 'Worker artifact differs')
    a, b = [pd.read_parquet(recovery.DIRECTORY/m/'assembled/feature-prefix.parquet') for m in ('study', 'replay')]
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    diagnostics = diagnostic_checks(read(recovery.DIRECTORY/'study/assembled/diagnostics.json'))
    require(read(recovery.DIRECTORY/'study/assembled/diagnostics.json') == read(study.DIRECTORY/'diagnostics-preread.json'),
            'Diagnostics drifted during historical calculation')
    features = feature_checks(a, pd.read_parquet(study.DIRECTORY/'ETF-input.parquet'),
        read(recovery.DIRECTORY/'study/assembled/summary.json'))
    inputs = read(study.DIRECTORY/'dependencies.json')
    require(inputs['original_models_found'] == dict.fromkeys(study.MODELS, []) and
            inputs['bars_1m_registered'] is False and inputs['strict_usable'] is False,
            'Missing input scope differs')
    guard = read(GUARD)
    require(guard['status'] == 'ok' and guard['skips'] == 0 and guard['local_passed'] ==
            guard['isolated_passed'] == 42 and all(c['returncode'] == 0 for c in guard['commands']),
            'Formal validation incomplete')
    for name, digest in {**guard['implementation_sha256'], **guard['evidence_sha256']}.items():
        if name == 'src/observe/strategy_catalog.py': continue
        require(file_sha(name) == digest, 'Formal validated bytes changed: '+name)
    isolation = read(GUARD.parent/'isolated/isolation.json'); copied_root = Path(isolation['cwd'])
    require(isolation['tests_passed'] == 42 and isolation['skips'] == 0 and
            len(isolation['actual_imports']) == 6 and len(isolation['copied_sha256']) == guard['copied_files'] and
            all(Path(p).is_relative_to(copied_root) for p in isolation['actual_imports'].values()),
            'Formal copied import/test scope differs')
    for name, digest in isolation['copied_sha256'].items():
        require(file_sha(copied_root/name) == digest, 'Formal copied bytes changed: '+name)
    for mode in ('local', 'isolated'):
        folder = GUARD.parent/mode; command = read(folder/'command.json')
        cases = list(ET.parse(folder/'tests.xml').getroot().iter('testcase'))
        require(command['returncode'] == 0 and len(cases) == 42 and all(len(c) == 0 for c in cases) and
                command['cwd'] == (str(Path.cwd()) if mode == 'local' else str(copied_root)),
                'Captured full test command/skip scope differs')
    return {'features': features, 'diagnostics': diagnostics, 'offline': 'match/0',
        'local_tests': guard['local_passed'], 'isolated_tests': guard['isolated_passed'],
        'new_trading_backtests': 0, 'original_strategy_complete': False}


def prepare():
    require(not DIRECTORY.exists() and not RECEIPT.exists(), 'Immutable formal output exists')
    result = components(); selected = reviews(); active = read('data/catalog/strategies/latest.json')
    require(active['catalog_id'] == read(study.PARENT)['progress']['catalog_id'], 'Parent catalog differs')
    old = read(Path(active['directory'])/'catalog.json')
    before = (study.DIRECTORY/'implementation/src/observe/strategy_catalog.py').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == before, 'Catalog changed before preparation')
    DIRECTORY.mkdir(parents=True); save(DIRECTORY/'source-reviews.json', selected)
    shutil.copyfile(study.DIRECTORY/'baseline.json', DIRECTORY/'baseline.json')
    REVIEWS.update(selected); digests = []
    for tag in ('candidate', 'candidate-replay'):
        root = DIRECTORY/tag; target = root/'catalog/strategies/implementation-evidence.json'
        target.parent.mkdir(parents=True); shutil.copyfile(study.DIRECTORY/'implementation-evidence.before.json', target)
        proposed = catalog_strategies(root, 'repo/量化策略源代码'); folder = Path(proposed['output'])
        candidate_checks(old, read(folder/'catalog.json'), selected)
        digests.append({n: file_sha(folder/n) for n in ('catalog.json', 'summary.json', 'catalog.parquet')})
        if tag == 'candidate': candidate = proposed
    require(digests[0] == digests[1], 'Catalog candidate byte replay differs')
    (DIRECTORY/'strategy_catalog.proposed.py').write_bytes(catalog_program(before, selected))
    paths = [*SPECS, *PHASES, str(GUARD), str(recovery.DIRECTORY/'offline-verification.json'),
        str(recovery.DIRECTORY/'prepared.json'), str(study.DIRECTORY/'prepared.json'),
        str(recovery.DIRECTORY/'study/assembled/summary.json'),
        str(DIRECTORY/'source-reviews.json'), 'scripts/accept_strategy_batch100.py',
        'tests/unit/test_batch100_acceptance.py']
    save(DIRECTORY/'prepared.json', {'components': result, 'candidate': candidate, 'catalog_bytes': digests[0],
        'previous_catalog': active, 'evidence_sha256': {n: file_sha(n) for n in paths},
        'proposed_code_sha256': file_sha(DIRECTORY/'strategy_catalog.proposed.py')})
    return {'status': 'candidate_verified', 'not_published': True, 'catalog_id': candidate['catalog_id']}


def publish():
    require(not RECEIPT.exists(), 'Immutable formal receipt exists')
    doc = read(DIRECTORY/'prepared.json'); selected = reviews()
    for name, digest in doc['evidence_sha256'].items(): require(file_sha(name) == digest, 'Prepared evidence changed')
    require(components() == doc['components'] and REPORT.is_file(), 'Formal result/report differs')
    before = (study.DIRECTORY/'implementation/src/observe/strategy_catalog.py').read_bytes()
    require(Path('src/observe/strategy_catalog.py').read_bytes() == catalog_program(before, selected) and
            file_sha('src/observe/strategy_catalog.py') == doc['proposed_code_sha256'] and
            all(REVIEWS.get(n) == v for n, v in selected.items()), 'Runtime proposed rules differ')
    candidate = Path(doc['candidate']['output'])
    candidate_checks(read(Path(doc['previous_catalog']['directory'])/'catalog.json'), read(candidate/'catalog.json'), selected)
    for name, digest in doc['catalog_bytes'].items(): require(file_sha(candidate/name) == digest, 'Candidate bytes changed')
    with operation_lock('data', DATA_WRITER):
        latest = Path('data/catalog/strategies/latest.json'); progress_latest = Path('data/catalog/strategies/progress/latest.json')
        require(read(latest) == doc['previous_catalog'], 'Active catalog pointer changed')
        require(Store('data').published()['batch_id'] == read(study.DIRECTORY/'dependencies.json')['published_batch'],
                'Locked actual data-root publication changed')
        old_latest, old_progress = latest.read_bytes(), progress_latest.read_bytes()
        proof = read(DIRECTORY/'protection.json')
        require(proof['status'] == 'ok' and proof['old_controls_unchanged'] == 6614 and
                proof['old_partition_metadata_unchanged'] == 398, 'Formal old-asset protection incomplete')
        dest = Path('data/catalog/strategies')/doc['candidate']['catalog_id']; dest.mkdir(exist_ok=False)
        for name, digest in doc['catalog_bytes'].items():
            shutil.copyfile(candidate/name, dest/name); require(file_sha(dest/name) == digest, 'Published bytes differ')
        try:
            _atomic_json(latest, {'catalog_id': dest.name, 'directory': str(dest), 'sha256': doc['catalog_bytes']['catalog.json']})
            progress = archive('data', 'Batch100 source research accepted:4852 original400-row feature prefixes/32 diagnostics/offline match0;model/minute/fund trading blocked')
            require((progress['manually_reviewed'], progress['locally_backtested_sources'],
                progress['locally_reproduced_sources'], progress['local_strategy_runs']) == (286, 12, 12, 87), 'Formal counts differ')
            bound = {**doc['evidence_sha256'], str(DIRECTORY/'prepared.json'): file_sha(DIRECTORY/'prepared.json'),
                str(REPORT): file_sha(REPORT), 'src/observe/strategy_catalog.py': file_sha('src/observe/strategy_catalog.py'),
                str(DIRECTORY/'protection.json'): file_sha(DIRECTORY/'protection.json'),
                str(dest/'catalog.json'): file_sha(dest/'catalog.json'),
                str(Path(progress['output'])/'summary.json'): file_sha(Path(progress['output'])/'summary.json')}
            require(read(latest)['catalog_id'] == progress['catalog_id'] == dest.name and
                    read(progress_latest)['checkpoint'] == progress['checkpoint'] and
                    all(file_sha(n) == digest for n, digest in bound.items()), 'Final catalog identity/navigation differs')
            save(RECEIPT, {'status': 'ok', 'formal_source_research_acceptance': True, 'formal_trading_acceptance': False,
                'library_goal_complete': False, 'original_strategy_complete': False, 'progress': progress,
                'components': doc['components'], 'source_sha256': list(study.SOURCE_SHA), 'evidence_sha256': bound,
                'protection': proof, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
                'phase_specs_remain_frozen': True, 'review_method': 'Full original rules, derived scalar tails, copied code tests and98 fresh network-blocked original-expression chunks; no independent agent review'})
        except Exception:
            latest.write_bytes(old_latest); progress_latest.write_bytes(old_progress); raise
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'reviewed': 286, 'remaining': 409}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'publish', 'protect')); args = parser.parse_args()
    result = protect('data', DIRECTORY) if args.action == 'protect' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False))
