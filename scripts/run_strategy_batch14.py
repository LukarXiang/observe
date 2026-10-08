"""Append selected historical inputs and extend approved frozen strategy variants."""
import argparse
from datetime import date
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from observe.data import standardize as std
from observe.data.audit import audit_daily
from observe.data.indices import audit_index, standardize_index
from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.execution import sessions
from observe.integrity import verify_run
from observe.ledger import RuleSet
from observe.runs import environment, file_sha, write_json
from observe.strategies import StrategyConfig, _signal_window
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch14 import END, INSTRUMENTS, SNAPSHOT, START, checked_artifacts
from scripts.run_strategy_batch4 import fee_checks, mutable_registry, pair_signal_check, read
from scripts.run_strategy_batch5 import run_one, signal_check as multi_check, verify_one
from scripts.run_strategy_batch6 import signal_check as svm_check
from scripts.run_strategy_batch8 import signal_check as rotation_check
from scripts.verify_strategy_batch3 import require, signal_check, tree_hash

CONFIGS = ('bollinger_long_batch4', 'ma10_ma20_long_batch4', 'ma5_ma10_long_batch4',
           'pair_yili_cmb_long_batch4', 'pair_haitian_cmb_long_batch4',
           'multi_ma_fixed_20k_batch5', 'multi_ma_fixed_1m_batch5',
           'svm_lagged_shape_batch6', 'pair_zscore_rotation_batch8')
CUTOFF = date(2021, 1, 1)
IMPLEMENTATION_FILES = {'scripts/probe_strategy_batch14.py', 'scripts/run_strategy_batch14.py',
    'src/observe/execution.py', 'src/observe/research.py', 'tests/unit/test_history_scope.py', 'tests/unit/test_batch14_gates.py',
    'src/observe/strategies.py', 'src/observe/strategy_means.py', 'tests/unit/test_strategy_means.py',
    'tests/integration/test_stable_strategy_means.py'}
CORE_EVIDENCE = {'existing-api.json', 'probe-results.json', 'retry-results.json', 'input-analysis.json',
    'publication.json', 'prepared-configs.json', 'historical-daily-audit.csv', 'fee-hand-checks.json',
    'numeric-policy-approval.json', 'stable-configs.json', 'numeric-resolution.json'}
PENDING_NUMERIC = {'ma5_ma10_long_batch4-extended': 'ma5-numeric-boundary-blocked.json',
    'multi_ma_fixed_20k_batch5-extended': 'multi_ma_fixed_20k_batch5-extended-numeric-boundary-blocked.json',
    'multi_ma_fixed_1m_batch5-extended': 'multi_ma_fixed_1m_batch5-extended-numeric-boundary-blocked.json'}


def retain_old_rows(old, current, keys):
    require(not current.duplicated(keys).any(), 'Current table has duplicate keys')
    retained = current.merge(old[keys], on=keys, how='inner', validate='one_to_one')
    pd.testing.assert_frame_equal(retained.sort_values(keys).reset_index(drop=True), old.sort_values(keys).reset_index(drop=True), check_dtype=False)


def check_references(old, current):
    for table, entries in old['tables'].items():
        if table in ('calendar', 'index_1d'): continue
        now = current['tables'].get(table, {})
        require(all(now.get(key) == value for key, value in entries.items()), f'Old {table} partition references changed')
        if table != 'bars_1d': require(now == entries, f'Unexpected changes to {table}')
    require(set(current['tables']) == set(old['tables']), 'Unexpected table added or removed')
    extra = set(current['tables']['bars_1d']) - set(old['tables']['bars_1d'])
    require(all(key.isdigit() and 2005 <= int(key) < 2021 for key in extra), 'New bar partitions must be disjoint pre-2021 years')
    return sorted(extra)


def extension_frames(store, state, artifacts):
    require(set(artifacts) == {'calendar', 'index_000300', *(f'{i}_{kind}' for i in INSTRUMENTS for kind in ('bars', 'factors'))}, 'Incomplete historical inputs')
    raw_calendar = artifacts['calendar'][1]
    require(set(raw_calendar.is_trading_day.astype(str)) <= {'0', '1'}, 'Unknown calendar flag')
    calendar = std.calendar(raw_calendar)
    require(not calendar.date.duplicated().any() and set(calendar.date) == set(pd.date_range(START, END).date), 'Incomplete calendar dates')
    old_calendar = store.load_state(state, 'calendar'); retain_old_rows(old_calendar, calendar, ['date'])
    index = standardize_index(artifacts['index_000300'][1]); days = sessions(calendar)
    require(not audit_index(index, '000300.SH', days), 'Historical index audit failed')
    old_index = store.load_state(state, 'index_1d'); retain_old_rows(old_index, index, ['date', 'index'])
    master = store.load_state(state, 'instruments'); listing = master.set_index('instrument').list_date
    filters = [('instrument', 'in', list(INSTRUMENTS))]
    factors = store.load_state(state, 'adj_factors', filters=filters)
    coverage = store.load_state(state, 'adj_coverage', filters=filters)
    frames = []
    for instrument in INSTRUMENTS:
        frame = artifacts[f'{instrument}_bars'][1]
        require(set(frame.tradestatus.astype(str)) <= {'0', '1'} and set(frame.isST.astype(str)) <= {'0', '1'}, 'Unknown stock flag')
        bars = std.daily(frame)
        require(set(bars.instrument) == {instrument} and not bars.duplicated(['date', 'instrument']).any(), 'Unexpected instrument or duplicate bars')
        require(set(bars.date) == {d for d in days if d >= listing.loc[instrument]}, f'{instrument}: missing or extra sessions')
        px = bars.loc[bars.is_trading, ['open', 'high', 'low', 'close', 'preclose']].to_numpy(float)
        require(np.isfinite(px).all() and (px > 0).all(), f'{instrument}: invalid trading prices')
        amounts = bars[['volume', 'amount']].to_numpy(float)
        require(np.isfinite(amounts).all() and (amounts >= 0).all(), f'{instrument}: invalid volume or amount')
        old = store.load_state(state, 'bars_1d', filters=[('instrument', '==', instrument)])
        retain_old_rows(old, bars, ['date', 'instrument'])
        fresh = artifacts[f'{instrument}_factors'][1]
        require(not fresh.duplicated(['code', 'dividOperateDate']).any(), 'Duplicate raw factor events')
        incoming = std.adj_factors(fresh)
        pd.testing.assert_frame_equal(incoming.reset_index(drop=True), factors[factors.instrument.eq(instrument)].reset_index(drop=True))
        adjusted = with_adjusted(bars, factors, coverage)
        require(adjusted.loc[bars.is_trading, 'close_adj'].notna().all(), f'{instrument}: adjustment coverage missing')
        prefix = bars[bars.date < CUTOFF].copy()
        require(len(prefix) > 0, 'No new historical bars')
        frames.append(prefix)
    bars = pd.concat(frames, ignore_index=True)
    problems = audit_daily(bars, [d for d in days if d < CUTOFF], master, RuleSet.from_yaml('configs/rule_profiles/main_board.yaml'))
    require(not problems.level.eq('block').any(), 'Historical daily audit contains blocking issues')
    return bars, pd.concat([calendar[calendar.date < old_calendar.date.min()], old_calendar], ignore_index=True), pd.concat([index[index.date < old_index.date.min()], old_index], ignore_index=True), problems


def publish(root, directory):
    output = directory / 'publication.json'
    if output.exists(): raise FileExistsError(output)
    analysis = read(directory / 'input-analysis.json'); artifacts, evidence = checked_artifacts(directory)
    require(analysis['snapshot'] == SNAPSHOT and analysis['probe_evidence'] == evidence, 'Analysis evidence differs from checked probes')
    store = Store(root); baseline = read(directory / 'baseline.json')
    with operation_lock(root, DATA_WRITER):
        state = store.published()
        require(state == baseline['published'] and state['tables'] == store.state(SNAPSHOT)['tables'], 'Published state moved since baseline')
        bars, calendar, index, issues = extension_frames(store, state, artifacts)
        issues_path = directory / 'historical-daily-audit.csv'; require(not issues_path.exists(), 'Audit already exists')
        issues.to_csv(issues_path, index=False)
        parts = {'bars_1d': {}, 'calendar': {'all': store.write_partition('calendar', 'all', calendar)},
                 'index_1d': {'all': store.write_partition('index_1d', 'all', index)}}
        for year, frame in bars.groupby(bars.date.map(lambda d: d.year)):
            key = str(year); require(key not in state['tables']['bars_1d'], 'Historical year already exists')
            entry = store.write_partition('bars_1d', key, frame)
            entry['history_scope'] = {'policy': 'selected_instruments_v1', 'instruments': list(INSTRUMENTS),
                'start': str(frame.date.min()), 'end': str(frame.date.max()), 'full_market': False,
                'probe_evidence': evidence}
            parts['bars_1d'][key] = entry
        prospective = {'tables': {**state['tables'], 'bars_1d': {**state['tables']['bars_1d'], **parts['bars_1d']},
                                 'calendar': parts['calendar'], 'index_1d': parts['index_1d']}}
        check_references(state, prospective)
        bid = store.write_batch(parts, note='batch14: pre-2021 selected ten-stock history only; old overlap retained; original factors/actions reused')
        rules = RuleSet.from_yaml('configs/rule_profiles/main_board.yaml')
        audit = store.commit_audit(bid, issues, rules.config_fingerprint(), [START, '2020-12-31'],
                                  scope='selected_history', input_state=prospective)
        store.publish(bid); sid = store.snapshot('batch14 selected ten-stock pre-2021 extension; not a complete historical market')
        result = {'status': 'published', 'batch_id': bid, 'snapshot': sid, 'old_snapshot': SNAPSHOT,
            'bars_added': len(bars), 'bar_partitions_added': len(parts['bars_1d']), 'calendar_rows': len(calendar),
            'index_rows': len(index), 'inputs': evidence, 'analysis_sha256': file_sha(directory / 'input-analysis.json'),
            'audit_id': audit, 'issues_file': str(issues_path), 'issues_sha256': file_sha(issues_path),
            'new_warning_counts': issues.groupby('rule').size().to_dict(), 'original_warnings_retained': 332,
            'historical_universe': list(INSTRUMENTS), 'full_historical_market': False,
            'old_adjustments_and_actions_reused': True, 'strict_financial_usable_rows': 0}
        write_json(output, result)
    archive(root, f'第十四批仅十股历史发布{bid}，新增{len(bars)}行；旧行情逐值保留，全市场历史仍未补齐')
    return result


def prepare(root, directory):
    destination = directory / 'configs'; destination.mkdir(exist_ok=False)
    publication = read(directory / 'publication.json'); store = Store(root); state = store.state(publication['snapshot'])
    days = sessions(store.load_state(state, 'calendar')); records = []
    fees = yaml.safe_load(Path('configs/rule_profiles/main_board.yaml').read_text())['fees']
    fee_start = min(pd.Timestamp(row['start']).date() for row in fees)
    for label in CONFIGS:
        original = Path('configs/strategies') / f'{label}.yaml'; cfg = yaml.safe_load(original.read_text(encoding='utf-8'))
        spec = StrategyConfig.model_validate(cfg); scope = cfg['execution_instruments']; filters = [('instrument', 'in', scope)]
        bars = store.load_state(state, 'bars_1d', filters=filters)
        warmup = max(_signal_window(spec), spec.execution.liquidity_window, spec.universe.min_listed_sessions,
                     spec.universe.suspend_window, spec.universe.liquidity_window)
        first = max(bars.groupby('instrument').date.min()); k = days.index(first) + warmup
        reasons = [{'kind': 'data_and_warmup', 'first_joint_day': str(first), 'warmup': warmup, 'earliest': str(days[k])}]
        actions = store.load_state(state, 'corp_actions', filters=filters)
        rights = actions[(actions.rights_ratio > 0) & (actions.ex_date >= days[k])]
        if len(rights):
            boundary = rights.ex_date.max(); k = max(k, next(n for n, d in enumerate(days) if d > boundary))
            reasons.append({'kind': 'rights_not_supported', 'last_event': str(boundary), 'policy': 'Start after last rights event; no subscription assumption'})
        if spec.implementation in ('svm_lagged_shape_corrected_v1', 'pair_zscore_rotation_corrected_v1'):
            fields = ['open', 'high', 'low', 'close', 'volume'] if spec.implementation.startswith('svm') else ['close']
            bad = bars[(~bars.is_trading) | (~np.isfinite(bars[fields]) | (bars[fields] <= 0)).any(axis=1)]
            if len(bad):
                boundary = bad.date.max(); k = max(k, days.index(boundary) + warmup + 1)
                reasons.append({'kind': 'strict_history_window', 'last_unusable_day': str(boundary),
                    'policy': 'Longest uninterrupted supported tail; no pause filling or feature changes'})
        while days[k] < fee_start: k += 1
        require(k < len(days) and days[k] <= date.fromisoformat(END), 'No supported extended interval')
        cfg.update(snapshot=publication['snapshot'], start=str(days[k]), cache=False)
        spec = StrategyConfig.model_validate(cfg)
        path = destination / f'{label}-extended.yaml'; path.write_text(yaml.safe_dump(spec.model_dump(mode='json'), allow_unicode=True, sort_keys=False), encoding='utf-8')
        records.append({'label': path.stem, 'config_file': str(path), 'config_sha256': file_sha(path),
            'original_config_file': str(original), 'original_config_sha256': file_sha(original),
            'source_sha256': file_sha(spec.source_path), 'start': str(spec.start), 'end': str(spec.end),
            'sessions': sum(spec.start <= d <= spec.end for d in days), 'boundary_reasons': reasons,
            'earliest_supported_fee_date': str(fee_start), 'parameters_changed': False})
    write_json(directory / 'prepared-configs.json', {'snapshot': publication['snapshot'], 'configs': records,
        'selection': 'Earliest data/rules-safe start, strict-window strategies use the continuous tail; no performance selection'})
    archive(root, '第十四批九份批准变体的历史延伸参数已冻结；按上市/预热/费用/配股/缺失窗口选择起点')
    return records


def config_record(directory, label):
    prepared = read(directory / 'prepared-configs.json')
    records = list(prepared['configs'])
    if (directory / 'stable-configs.json').exists(): records += read(directory / 'stable-configs.json')['configs']
    record = next(r for r in records if r['label'] == label)
    require(file_sha(Path(record['config_file'])) == record['config_sha256'], 'Prepared config changed')
    cfg = yaml.safe_load(Path(record['config_file']).read_text(encoding='utf-8'))
    require(file_sha(Path(cfg['source_path'])) == record['source_sha256'], 'Prepared source changed')
    require(file_sha(Path(record['original_config_file'])) == record['original_config_sha256'], 'Original config changed')
    if 'parent_label' in record:
        require(file_sha(directory / 'numeric-policy-approval.json') == record['approval_sha256'], 'Numeric approval changed')
        parent = config_record(directory, record['parent_label'])
        require(parent['config_sha256'] == record['parent_config_sha256'], 'Parent config changed')
        require(file_sha(directory / PENDING_NUMERIC[record['parent_label']]) == record['diagnosis_sha256'], 'Numeric diagnosis changed')
    return record


def prepare_stable(root, directory):
    destination = directory / 'stable-configs'; destination.mkdir(exist_ok=False)
    approval_path = directory / 'numeric-policy-approval.json'
    require(not approval_path.exists(), 'Numeric approval already exists')
    write_json(approval_path, {'status': 'approved', 'policy': 'window_fsum_v1',
        'user_answers': ['推进数值边界修正变体（推荐）', '同步推进复星数值修正变体（推荐）'],
        'labels': sorted(PENDING_NUMERIC), 'rules': 'Per-window math.fsum then binary division; strict comparisons, no tolerance; original parameters and cash retained',
        'platform_equivalence_proven': False})
    records = []
    for label, diagnostic in PENDING_NUMERIC.items():
        parent = config_record(directory, label); cfg = yaml.safe_load(Path(parent['config_file']).read_text(encoding='utf-8'))
        cfg['parameters']['mean_algorithm'] = 'window_fsum_v1'
        cfg['name'] += '（逐窗口稳定求和修正）'
        spec = StrategyConfig.model_validate(cfg); path = destination / f'{label}-stable.yaml'
        path.write_text(yaml.safe_dump(spec.model_dump(mode='json'), allow_unicode=True, sort_keys=False), encoding='utf-8')
        records.append({**parent, 'label': path.stem, 'config_file': str(path), 'config_sha256': file_sha(path),
            'parent_label': label, 'parent_config_sha256': parent['config_sha256'],
            'approval_sha256': file_sha(approval_path), 'diagnosis_sha256': file_sha(directory / diagnostic),
            'parameters_changed': True, 'only_parameter_change': {'mean_algorithm': 'window_fsum_v1'}})
    write_json(directory / 'stable-configs.json', {'configs': records, 'selection': 'User-approved numerical correction, no performance selection'})
    archive(root, '第十四批国航与复星两种资金的数值修正批准已存档，三份新配置单独冻结')
    return records


def decimal_mean(values):
    if not all(math.isfinite(v) for v in values): return math.nan
    with localcontext() as context:
        context.prec = 1200
        total = sum((Decimal.from_float(float(v)) for v in values), Decimal(0))
        return float(Decimal.from_float(float(total)) / len(values))


def stable_check(root, output):
    cfg = read(output / 'config.json')['config']; require(cfg['parameters']['mean_algorithm'] == 'window_fsum_v1', 'Wrong numerical policy')
    store = Store(root); state = store.state(cfg['snapshot']); filters = [('instrument', 'in', cfg['execution_instruments'])]
    view = with_adjusted(store.load_state(state, 'bars_1d', filters=filters), store.load_state(state, 'adj_factors', filters=filters),
                         store.load_state(state, 'adj_coverage', filters=filters))
    close = view.set_index('date').close_adj.where(view.set_index('date').is_trading).reindex(sessions(store.load_state(state, 'calendar')))
    windows = {f'ma{n}': n for n in (5, 10, 20, 30)} if cfg['implementation'] == 'multi_ma_fixed_value_v1' else {'ma_short': 5, 'ma_long': 10}
    checked = 0
    for row in pd.read_parquet(output / 'factors.parquet').itertuples():
        k = close.index.get_loc(row.date)
        for column, n in windows.items():
            value = decimal_mean(close.iloc[k - n + 1:k + 1])
            actual = getattr(row, column)
            require((math.isnan(value) and pd.isna(actual)) or value == actual, f'{row.date}: exact Decimal {column} mismatch')
            checked += 1
    proof = (multi_check if cfg['implementation'] == 'multi_ma_fixed_value_v1' else signal_check)(root, output)
    return {**proof, 'exact_decimal_windows_checked': checked, 'numerical_policy': 'window_fsum_v1',
        'numerical_method': 'Exact Decimal sum of binary inputs, rounded binary sum then division; equality checked without tolerance'}


def check_verified_stable(record, output, verified):
    expected = StrategyConfig.model_validate(yaml.safe_load(Path(record['config_file']).read_text(encoding='utf-8'))).model_dump(mode='json')
    require(read(output / 'config.json')['config'] == expected and expected['parameters'].get('mean_algorithm') == 'window_fsum_v1', 'Stable actual config differs')
    count = record['sessions']; windows = 4 if expected['implementation'] == 'multi_ma_fixed_value_v1' else 2
    proof = verified['hand_check']
    require(proof.get('numerical_policy') == 'window_fsum_v1' and proof.get('sessions_checked') == count and
            proof.get('exact_decimal_windows_checked') == count * windows, 'Stable Decimal coverage incomplete')


def verify(root, directory, label):
    record = config_record(directory, label); spec = yaml.safe_load(Path(record['config_file']).read_text(encoding='utf-8'))
    implementation = spec['implementation']
    checker = {'multi_ma_fixed_value_v1': multi_check, 'svm_lagged_shape_corrected_v1': svm_check,
               'pair_zscore_rotation_corrected_v1': rotation_check}.get(implementation,
                pair_signal_check if implementation.startswith('pair_') else signal_check)
    if spec['parameters'].get('mean_algorithm') == 'window_fsum_v1': checker = stable_check
    return verify_one(root, directory, label, checker=checker, batch_label='第十四批')


def compare_stable(root, directory):
    destination = directory / 'numeric-resolution.json'; require(not destination.exists(), 'Numeric resolution already exists')
    records = read(directory / 'stable-configs.json')['configs']; results = []
    for record in records:
        config_record(directory, record['label'])
        verified = read(directory / f'{record["label"]}-verification.json')
        parent = read(directory / f'{record["parent_label"]}-run.json'); current = verified['run']
        paths = []
        for run in (parent, current):
            integrity = verify_run(root, run['run_id']); require(integrity['status'] == 'ok', 'Comparison run integrity failed')
            path = Path(integrity['output']); require(path.resolve() == Path(run['output']).resolve(), 'Comparison run output mismatch')
            paths.append(path)
        old, new = paths; check_verified_stable(record, new, verified)
        old_cfg, new_cfg = (read(path / 'config.json')['config'] for path in paths)
        comparable = {**new_cfg, 'name': old_cfg['name'], 'parameters': {k: v for k, v in new_cfg['parameters'].items() if k != 'mean_algorithm'}}
        require(comparable == old_cfg, 'Stable variant changed economic configuration')
        left, right = (pd.read_parquet(path / 'factors.parquet').set_index('date') for path in paths)
        require(left.index.equals(right.index), 'Comparison dates changed')
        columns = ['bull', 'bear', 'struggle', 'crossup', 'crossdown'] if 'bull' in left else ['state']
        changed = (left[columns] != right[columns]).any(axis=1)
        ltargets, rtargets = (pd.read_parquet(path / 'targets.parquet') for path in paths)
        require(ltargets[['decision_date', 'instrument']].equals(rtargets[['decision_date', 'instrument']]), 'Comparison target keys differ')
        target_columns = ['enter_when_empty', 'exit', 'skip_when_empty'] if 'bull' in left else ['weight']
        target_changed = (ltargets[target_columns] != rtargets[target_columns]).any(axis=1)
        scenarios = []
        for scenario in ('base', 'fees_x2', 'slippage_x2'):
            rows = []
            for path in paths:
                report = read(path / 'report.json'); child = next(r for r in read(path / 'subruns.json')['backtests'] if r['scenario'] == scenario)
                folder = Path(child['output']); result = next(r for r in report['results'] if r['scenario'] == scenario)
                rows.append({'run_id': child['run_id'], 'orders': len(read(folder / 'orders.json')),
                    'fills': len(read(folder / 'fills.json')), 'metrics': result['metrics']})
            scenarios.append({'scenario': scenario, 'original': rows[0], 'stable': rows[1],
                'total_return_change': rows[1]['metrics']['total_return'] - rows[0]['metrics']['total_return']})
        results.append({'label': record['label'], 'parent_label': record['parent_label'],
            'original_run_id': parent['run_id'], 'stable_run_id': current['run_id'],
            'original_manifest_sha256': file_sha(old / 'manifest.json'), 'stable_manifest_sha256': file_sha(new / 'manifest.json'),
            'source_sha256': record['source_sha256'], 'only_numerical_policy_changed': True,
            'factor_flag_or_state_difference_days': [str(d) for d in left.index[changed]],
            'target_intent_difference_days': sorted(set(str(d) for d in ltargets.loc[target_changed, 'decision_date'])),
            'cost_scenarios': scenarios})
    write_json(destination, {'status': 'ok', 'approval_sha256': file_sha(directory / 'numeric-policy-approval.json'),
        'original_independent_validation_passed': False, 'platform_float_equivalence_proven': False, 'variants': results})
    archive(root, '第十四批三份数值修正与原实验的逐日差异、订单/成交和三成本结果对照已存档')
    return {'status': 'ok', 'output': str(destination), 'variants': len(results)}


def checks(root, directory):
    destination = directory / 'checks.json'; require(not destination.exists(), 'Checks already exist')
    commands = [[sys.executable, '-m', 'pytest', '-q', 'tests/unit/test_batch14_gates.py', 'tests/unit/test_history_scope.py',
        'tests/unit/test_strategy_means.py', 'tests/integration/test_stable_strategy_means.py', 'tests/integration/test_strategy_scope.py',
        'tests/integration/test_multi_ma_strategy.py'],
        [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py',
         'scripts/build_strategy_catalog.py', 'scripts/verify_financial_import.py', 'scripts/probe_strategy_batch14.py', 'scripts/run_strategy_batch14.py'],
        ['git', 'diff', '--check']]
    results = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        results.append({'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        require(result.returncode == 0, 'Final verification command failed')
    evidence = {p.name: file_sha(p) for p in directory.iterdir() if p.is_file()}
    require(CORE_EVIDENCE <= set(evidence), 'Core evidence incomplete')
    write_json(destination, {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(IMPLEMENTATION_FILES)},
        'evidence_sha256': evidence, 'specialists': 'Architecture reviewed stable additions without findings; security configuration-binding finding fixed and regression tested'})
    return {'status': 'ok', 'output': str(destination)}


def diagnose_multi(root, directory, label):
    require(label in PENDING_NUMERIC and label.startswith('multi_ma'), 'Diagnostic only supports frozen multi-MA variants')
    destination = directory / PENDING_NUMERIC[label]
    if destination.exists(): raise FileExistsError(destination)
    record = config_record(directory, label); run = read(directory / f'{label}-run.json')
    integrity = verify_run(root, run['run_id']); require(integrity['status'] == 'ok', 'Diagnostic run integrity failed')
    output = Path(integrity['output']); require(Path(run['output']).resolve() == output.resolve(), 'Diagnostic run output mismatch')
    before = tree_hash(output); cfg = read(output / 'config.json')['config']
    store = Store(root); state = store.state(cfg['snapshot']); filters = [('instrument', 'in', cfg['execution_instruments'])]
    view = with_adjusted(store.load_state(state, 'bars_1d', filters=filters), store.load_state(state, 'adj_factors', filters=filters),
                         store.load_state(state, 'adj_coverage', filters=filters))
    close = view.set_index('date').close_adj.where(view.set_index('date').is_trading).reindex(sessions(store.load_state(state, 'calendar')))
    factors = pd.read_parquet(output / 'factors.parquet'); targets = pd.read_parquet(output / 'targets.parquet').set_index('decision_date')
    cases = []; intent_differences = 0
    bull = lambda a: a[0] > a[1] > a[2] > a[3]
    bear = lambda a: a[0] < a[1] < a[2]
    for row in factors.itertuples():
        k = close.index.get_loc(row.date)
        averages = lambda j: [math.fsum(close.iloc[j - n + 1:j + 1]) / n for n in (5, 10, 20, 30)]
        a, b, c = averages(k), averages(k - 1), averages(k - 2)
        stored = [getattr(row, f'ma{n}') for n in (5, 10, 20, 30)]
        require(all((math.isnan(x) and pd.isna(y)) or math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-8)
                    for x, y in zip(a, stored, strict=True)), 'Non-boundary mean mismatch requires separate diagnosis')
        flags = {'bull': bull(a), 'bear': bear(a), 'struggle': abs(a[1] / a[2] - 1) < .003 or abs(a[2] / a[3] - 1) < .002,
            'crossdown': bull(b) and bull(c) and b[0] > b[1] and a[0] < a[1], 'crossup': bear(b) and bear(c) and b[1] < b[2] and a[1] > a[2]}
        stored_flags = {name: bool(getattr(row, name)) for name in flags}
        if flags == stored_flags: continue
        intents = {'enter_when_empty': bool(row.valid and ((flags['bull'] and not flags['struggle']) or flags['crossup'])),
                   'exit': bool(row.valid and (flags['bear'] or flags['crossdown'])),
                   'skip_when_empty': bool(row.valid and flags['bull'] and flags['struggle'])}
        actual = targets.loc[row.date]; stored_intents = {key: bool(actual[key]) for key in intents}
        intent_differences += int(intents != stored_intents)
        cases.append({'date': str(row.date), 'window32_dates': [str(d) for d in close.index[k - 31:k + 1]],
            'window32_adjusted_closes': close.iloc[k - 31:k + 1].tolist(), 'fsum_ma': a, 'stored_ma': stored,
            'previous_fsum_ma': b, 'previous2_fsum_ma': c, 'fsum_flags': flags, 'stored_flags': stored_flags,
            'fsum_intents': intents, 'stored_intents': stored_intents,
            'float_hex': {'fsum': [float(x).hex() for x in a], 'stored': [float(x).hex() for x in stored]}})
    require(cases and tree_hash(output) == before, 'No numeric discrepancy found or original run changed')
    result = {'status': 'blocked_pending_numeric_policy', 'run_id': run['run_id'], 'source_sha256': record['source_sha256'],
        'original_manifest_sha256': file_sha(output / 'manifest.json'), 'original_unchanged': True, 'reproduction_claimed': False,
        'comparison_difference_count': len(cases), 'signal_difference_count': intent_differences, 'cases': cases,
        'cause': 'Tiny rolling/window-sum mean differences change strict source flags; no tolerance or economic fix applied',
        'confirmation': 'Pending user numerical-policy choice; old experiment remains immutable'}
    write_json(destination, result); archive(root, f'第十四批{label}浮点边界已冻结，独立验收未通过，数值政策待确认')
    return {'output': str(destination), 'comparison_difference_count': len(cases), 'signal_difference_count': intent_differences}


def protect(root, directory):
    destination = directory / 'protection.json'
    if destination.exists(): raise FileExistsError(destination)
    baseline = read(directory / 'baseline.json'); store = Store(root); current = store.published()
    controls = {name: sha for name, sha in baseline['controls'].items() if not mutable_registry(name)}
    require(all(file_sha(store.root / name) == sha for name, sha in controls.items()), 'Old controls changed')
    require(all((store.root / name).stat().st_size == item['size'] and (store.root / name).stat().st_mtime_ns == item['mtime_ns']
                for name, item in baseline['partitions'].items()), 'Old partition metadata changed')
    added = check_references(baseline['published'], current)
    for key in added:
        entry = current['tables']['bars_1d'][key]; frame = pd.read_parquet(store.root / entry['file'])
        require(set(frame.instrument) <= set(INSTRUMENTS) and frame.date.map(lambda d: d.year == int(key) and d < CUTOFF).all(), 'Unexpected historical prefix')
        require(entry['history_scope']['instruments'] == list(INSTRUMENTS), 'Selected scope metadata changed')
    for table, keys in [('calendar', ['date']), ('index_1d', ['date', 'index'])]:
        retain_old_rows(store.load_state(baseline['published'], table), store.load_state(current, table), keys)
    result = {'status': 'ok', 'old_controls_unchanged': len(controls), 'old_partition_metadata_unchanged': len(baseline['partitions']),
        'new_disjoint_bar_partitions': added, 'old_calendar_rows_unchanged': len(store.load_state(baseline['published'], 'calendar')),
        'old_index_rows_unchanged': len(store.load_state(baseline['published'], 'index_1d')),
        'old_bar_partition_entries_unchanged': True, 'partition_check': 'size/mtime only; no fresh full old partition hash',
        'baseline_sha256': file_sha(directory / 'baseline.json')}
    write_json(destination, result); return result


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch14-verification.json')
    if output.exists(): raise FileExistsError(output)
    checks = read(directory / 'checks.json')
    require(checks['commands'] and all(c['returncode'] == 0 for c in checks['commands']), 'Verification commands did not pass')
    require(IMPLEMENTATION_FILES <= set(checks['implementation_sha256']), 'Missing required implementation checks')
    require(CORE_EVIDENCE <= set(checks['evidence_sha256']), 'Missing required core evidence checks')
    require(all(file_sha(Path(name)) == sha for name, sha in checks['implementation_sha256'].items()), 'Checked implementation changed')
    require(all(file_sha(directory / name) == sha for name, sha in checks['evidence_sha256'].items()), 'Checked evidence changed')
    checked_artifacts(directory)
    publication = read(directory / 'publication.json'); store = Store(root)
    require(store.published()['batch_id'] == publication['batch_id'], 'Published batch changed')
    require(store.state(publication['snapshot'])['tables'] == store.published()['tables'], 'Snapshot tables differ')
    prepared = read(directory / 'prepared-configs.json'); strategies = []
    require(len(prepared['configs']) == len(CONFIGS) and {r['label'] for r in prepared['configs']} == {f'{label}-extended' for label in CONFIGS}, 'Prepared strategy set incomplete')
    stable_path = directory / 'stable-configs.json'
    stable_records = read(stable_path)['configs'] if stable_path.exists() else []
    for record in [*prepared['configs'], *stable_records]:
        config_record(directory, record['label'])
        run = read(directory / f'{record["label"]}-run.json')
        needed = {f'{record["label"]}-run.json', f'{record["label"]}-freeze.json'}
        proof_path = directory / f'{record["label"]}-verification.json'
        numeric_pending = record['label'] in PENDING_NUMERIC and run['status'] == 'success_limited' and not proof_path.exists()
        if numeric_pending: needed.add(PENDING_NUMERIC[record['label']])
        elif run['status'] == 'success_limited': needed.add(proof_path.name)
        require(needed <= set(checks['evidence_sha256']), 'Missing required strategy evidence checks')
        require(file_sha(Path(record['original_config_file'])) == record['original_config_sha256'], 'Original config changed')
        if numeric_pending:
            diagnostic = directory / PENDING_NUMERIC[record['label']]; pending = read(diagnostic)
            differences = pending.get('comparison_difference_count', pending.get('state_difference_count', 0))
            integrity = verify_run(root, run['run_id'])
            require(integrity['status'] == 'ok', 'Pending numeric run integrity failed')
            run_path = Path(integrity['output'])
            require(Path(run['output']).resolve() == run_path.resolve(), 'Pending numeric run output mismatch')
            require(pending['status'] == 'blocked_pending_numeric_policy' and pending['run_id'] == run['run_id'] and
                    pending['source_sha256'] == record['source_sha256'] and pending['original_unchanged'] and
                    not pending['reproduction_claimed'] and differences > 0 and
                    pending['original_manifest_sha256'] == file_sha(run_path / 'manifest.json'), 'Numeric diagnosis does not bind original run')
            strategies.append({**record, 'run_id': run['run_id'], 'status': 'blocked_pending_numeric_policy',
                'run_status': run['status'], 'reproduction': None, 'original_strategy_complete': False,
                'independent_validation_passed': False, 'diagnosis_file': str(diagnostic), 'diagnosis_sha256': file_sha(diagnostic),
                'independent_difference_count': differences, 'results': read(run_path / 'report.json')['results']})
        elif run['status'] == 'success_limited':
            path = proof_path; verified = read(path)
            require(verified['status'] == 'ok' and verified['run'] == run and verified['original_unchanged'] and
                    verified['source_sha256'] == record['source_sha256'] and
                    verified['reproduction']['reproduction']['result'] == 'match' and
                    verified['reproduction']['reproduction']['differences'] == 0, 'Strategy verification incomplete')
            integrity = verify_run(root, run['run_id'])
            require(integrity['status'] == 'ok', 'Final original run integrity failed')
            repeated = verify_run(root, verified['reproduction']['run_id'])
            require(repeated['status'] == 'ok', 'Final reproduction integrity failed')
            if 'parent_label' in record:
                check_verified_stable(record, Path(integrity['output']), verified)
                check_verified_stable(record, Path(repeated['output']), verified)
                require(read(Path(integrity['output']) / 'report.json') == verified['report'], 'Stable verified report differs')
            strategies.append({**record, 'run_id': run['run_id'], 'status': run['status'],
                'reproduction_run_id': verified['reproduction']['run_id'], 'reproduction': 'match', 'differences': 0,
                'hand_check': verified['hand_check'], 'period': verified['report']['period'],
                'results': verified['report']['results'], 'benchmark': verified['report']['benchmark'],
                'verification_file': str(path), 'verification_sha256': file_sha(path), 'original_strategy_complete': False})
        else:
            require(run['status'] == 'blocked', 'Nonterminal or failed strategy requires investigation')
            strategies.append({**record, 'run_id': run['run_id'], 'status': 'blocked', 'reproduction': None, 'original_strategy_complete': False})
    require({r['label'] for r in stable_records} == {f'{label}-stable' for label in PENDING_NUMERIC} and len(stable_records) == 3, 'Stable strategy set incomplete')
    by_label = {r['label']: r for r in strategies}
    for label in PENDING_NUMERIC:
        parent, corrected = by_label[label], by_label[f'{label}-stable']
        require(parent['status'] == 'blocked_pending_numeric_policy' and corrected.get('reproduction') == 'match', 'Numeric resolution incomplete')
        parent.update(status='superseded_numeric_validation_failed', replacement_label=corrected['label'],
                      replacement_run_id=corrected['run_id'], approval_sha256=file_sha(directory / 'numeric-policy-approval.json'))
    fees = read(directory / 'fee-hand-checks.json'); require(fees['status'] == 'ok', 'Fee checks missing')
    expected_costs = {(row['run_id'], scenario) for row in strategies for scenario in ('base', 'fees_x2', 'slippage_x2')}
    require({(row['run_id'], row['scenario']) for row in fees['checks']} == expected_costs and
            len(fees['checks']) == len(expected_costs), 'Fee scenario coverage incomplete')
    resolution = read(directory / 'numeric-resolution.json')
    require(resolution['status'] == 'ok' and resolution['approval_sha256'] == file_sha(directory / 'numeric-policy-approval.json') and
            {(r['parent_label'], r['original_run_id'], r['label'], r['stable_run_id']) for r in resolution['variants']} ==
            {(label, by_label[label]['run_id'], f'{label}-stable', by_label[f'{label}-stable']['run_id']) for label in PENDING_NUMERIC}, 'Numeric resolution binding incomplete')
    protection = protect(root, directory)
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for name, sha in checks['implementation_sha256'].items():
        target = frozen / Path(name).name; require(not target.exists(), 'Frozen implementation name collision')
        shutil.copyfile(name, target); require(file_sha(target) == sha, 'Frozen implementation differs')
    pending_count = sum(row['status'] == 'blocked_pending_numeric_policy' for row in strategies)
    result = {'status': 'archived_with_pending' if pending_count else 'ok', 'pending_numeric_policies': pending_count,
        'environment': environment(), 'publication': publication, 'strategies': strategies,
        'protection': protection, 'fee_checks': fees, 'checks': checks, 'numeric_resolution': resolution,
        'progress': archive(root, '第十四批历史延伸已归档；保留数据边界、三成本、独立核算、禁网复现与旧资产保护证据'),
        'limitations': ['Selected ten-stock history only; global and uncovered-scope loads blocked, including warmup',
            'Historical fee profiles remain externally unverified; source-platform equality and corporate-action date assumptions remain approximate',
            'Old 332 warnings retained, additional selected-history warnings archived; not a full snapshot daily audit',
            'Rights subscription is not implemented; CMB pairs start after its last rights event',
            'SVM and rotation keep strict complete windows, use continuous supported tails without pause filling',
            'No final holdout or parameter selection; strict financial usable rows remain zero']}
    write_json(output, result)
    return {'status': result['status'], 'output': str(output), 'strategies': len(strategies), 'pending_numeric_policies': pending_count,
            'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['publish', 'prepare', 'prepare-stable', 'run', 'verify', 'diagnose', 'fees', 'compare-stable', 'checks', 'protect', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch14/20261005-long-history'))
    parser.add_argument('--label')
    args = parser.parse_args()
    if args.action in ('run', 'verify', 'diagnose'):
        if not args.label: parser.error('--label required')
        record = config_record(args.directory, args.label)
        if args.action == 'run': result = run_one(args.root, args.directory, record['config_file'], batch_label='第十四批')
        else: result = {'verify': verify, 'diagnose': diagnose_multi}[args.action](args.root, args.directory, args.label)
    elif args.action == 'fees': result = fee_checks(args.directory)
    else: result = {'publish': publish, 'prepare': prepare, 'prepare-stable': prepare_stable, 'compare-stable': compare_stable,
                   'checks': checks, 'protect': protect, 'finish': finish}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
