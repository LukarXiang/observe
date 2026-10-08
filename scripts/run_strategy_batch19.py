"""Publish verified EMA55 histories, run approved approximation, archive every phase."""
import argparse
from datetime import date
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
from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.prices import with_adjusted
from observe.data.store import Store
from observe.execution import build, load, sessions
from observe.integrity import verify_run
from observe.ledger import Book, RuleSet
from observe.runs import canonical, compare_tables, environment, file_sha
from observe.strategies import StrategyConfig
from observe.strategy_ema import ANCHOR, PERIODS, POOL
from scripts.archive_strategy_progress import archive
from scripts.probe_strategy_batch14 import equal_cells
from scripts.review_strategy_batch18 import END, SNAPSHOT, STOCKS, artifacts, missing_stock_components, raw_quality, save
from scripts.run_strategy_batch4 import checkpoint, fee_checks, read
from scripts.run_strategy_batch5 import run_one, verify_one
from scripts.run_strategy_batch14 import retain_old_rows
from scripts.run_strategy_batch15 import merge_year, reproduce_old
from scripts.verify_strategy_batch3 import require

INPUTS = Path('data/staging/strategies-batch18/20261005-kd-ema-lof')
RECEIPT = Path('docs/handoff/2026-10-05-batch18-verification.json')
CONFIG = Path('configs/strategies/ema_slots_talib_batch19.yaml')
LABELS = ('ema55-short', 'ema55-long')
FILES = {'src/observe/strategy_ema.py', 'src/observe/strategies.py', 'src/observe/portfolio.py', 'src/observe/execution.py',
    'scripts/run_strategy_batch19.py', 'tests/integration/test_ema_strategy.py', 'tests/unit/test_history_scope.py',
    'tests/unit/test_batch19_gates.py', str(CONFIG)}
CORE = {'baseline.json', 'precheck.json', 'publication.json', 'prepared-configs.json', 'old-rsi-reproduction.json',
    'historical-daily-audit.csv', 'historical-daily-audit-v2.csv', 'publication-attempt-01.json', 'fee-hand-checks.json'} | {f'{label}-{suffix}.json' for label in LABELS for suffix in ('run', 'freeze', 'verification')}


def custody(directory):
    pre = read(directory / 'precheck.json')
    for path, sha in pre['bindings'].items(): require(file_sha(path) == sha, 'Bound batch18/config/source changed')
    receipt = read(RECEIPT)
    require(receipt['status'] == 'ok' and receipt['validated_analysis']['status'] == 'ready', 'Batch18 not ready')
    for path, sha in receipt['checks']['evidence_sha256'].items(): require(file_sha(INPUTS / path) == sha, 'Batch18 evidence changed')
    require(receipt['validated_analysis'] == read(INPUTS / 'input-validation-supplement.json'), 'Input supplement differs')
    return pre


def start(root, directory):
    checkpoint(root, directory, 'Batch19 approved EMA55 implementation baseline established; historical inputs reused with SHA binding')
    cfg = StrategyConfig.model_validate(yaml.safe_load(CONFIG.read_text(encoding='utf-8')))
    approval = read(INPUTS / 'ema-policy-approval.json')
    require(approval['status'] == 'approved_by_user' and approval['variant'] == cfg.implementation, 'EMA policy not approved')
    require(file_sha(cfg.source_path) == approval['source_sha256'], 'Approved EMA source differs')
    require(Store(root).published()['tables'] == Store(root).state(SNAPSHOT)['tables'], 'Pre-publication snapshot moved')
    paths = [RECEIPT, CONFIG, INPUTS / 'ema-policy-approval.json', INPUTS / 'input-validation-supplement.json', Path(cfg.source_path)]
    save(directory / 'precheck.json', {'bindings': {str(p): file_sha(p) for p in paths}, 'approval': approval,
        'old_snapshot': SNAPSHOT, 'method': 'Earliest complete current/prior EMA60 and liquidity; short engineering then full available period, no return selection',
        'reference': {'file': 'repo/backtrader/backtrader/indicators/basicops.py', 'sha256': file_sha('repo/backtrader/backtrader/indicators/basicops.py'),
            'use': 'ExponentialSmoothing SMA seed and recursive alpha reference; production uses locked TA-Lib'},
        'limitations': ['Original EMA platform not equivalent', 'Ex-post fixed pool', 'Historical partial-market partitions only', 'Existing fee/trading rule approximations retained']})
    custody(directory)
    return {'status': 'ok', 'directory': str(directory)}


def incoming_history(root, directory):
    custody(directory); frames = artifacts(INPUTS); store = Store(root); state = store.state(SNAPSHOT)
    filters = [('instrument', 'in', list(POOL))]
    old = store.load_state(state, 'bars_1d', filters=filters)
    factors = store.load_state(state, 'adj_factors', filters=filters); coverage = store.load_state(state, 'adj_coverage', filters=filters)
    master = store.load_state(state, 'instruments', filters=filters).set_index('instrument')
    calendar = sessions(store.load_state(state, 'calendar')); prefixes = []
    for instrument in STOCKS:
        require(not missing_stock_components(frames, instrument), 'Downloaded stock components incomplete')
        raw_bars = frames[f'{instrument}_bars']; quality = raw_quality(raw_bars)
        require(not any(quality['invalid_raw_flags'].values()) and not quality['invalid_flow_cells'], 'Invalid raw flags or flows')
        bars = std.daily(raw_bars); before = old[old.instrument.eq(instrument)]
        expected = [d for d in calendar if max(ANCHOR, master.loc[instrument, 'list_date']) <= d <= date.fromisoformat(END)]
        require(list(bars.date) == expected and set(bars.instrument) == {instrument}, 'Downloaded history dates/scope differ')
        overlap = before.merge(bars, on=['date', 'instrument'], validate='one_to_one', suffixes=('_old', '_new'))
        columns = list(before.columns.difference(['date', 'instrument']))
        require(len(overlap) == len(before) and not any(equal_cells(overlap[[f'{c}_old' for c in columns]].set_axis(columns, axis=1),
            overlap[[f'{c}_new' for c in columns]].set_axis(columns, axis=1), columns).values()), 'Old daily overlap differs')
        new_factors = std.adj_factors(frames[f'{instrument}_factors']).sort_values('ex_date').reset_index(drop=True)
        frozen = factors[factors.instrument.eq(instrument)][new_factors.columns].sort_values('ex_date').reset_index(drop=True)
        pd.testing.assert_frame_equal(new_factors, frozen, check_dtype=False, check_exact=True)
        prices = with_adjusted(bars, factors, coverage).loc[lambda r: r.is_trading, ['open', 'high', 'low', 'close', 'preclose', 'close_adj']].to_numpy(float)
        require(np.isfinite(prices).all() and (prices > 0).all(), 'Invalid trading prices/adjustments')
        joined = bars.merge(before[['date', 'instrument']], on=['date', 'instrument'], how='left', indicator=True, validate='one_to_one')
        prefixes.append(joined.loc[joined._merge.eq('left_only'), bars.columns])
    prefix = pd.concat(prefixes, ignore_index=True)
    require(len(prefix) == 19560 and prefix.date.lt(date(2021, 1, 1)).all(), 'Unexpected historical additions')
    return prefix


def publication_references(before, after, years):
    require(set(before['tables']) == set(after['tables']), 'Table set changed')
    for table, entries in before['tables'].items():
        now = after['tables'][table]
        if table != 'bars_1d': require(now == entries, f'Unexpected {table} change')
        else:
            require(set(entries) == set(now), 'Year keys changed')
            require({y for y in entries if entries[y] != now[y]} == set(years), 'Unexpected yearly references changed')


def annual_scope(scope, merged, master):
    instruments_ = sorted(set(merged.instrument)); missing = set(scope['instruments']) - set(instruments_)
    listings = master.set_index('instrument').list_date.to_dict(); boundary = max(merged.date)
    require(all(i in listings and pd.notna(listings[i]) and listings[i] > boundary for i in missing), 'Listed or unknown old scope lost')
    return instruments_, sorted(missing)


def publish(root, directory):
    require(not (directory / 'publication.json').exists(), 'Publication receipt exists')
    prefix = incoming_history(root, directory); years = sorted({str(d.year) for d in prefix.date}); store = Store(root)
    with operation_lock(root, DATA_WRITER):
        state = store.published(); require(state == read(directory / 'baseline.json')['published'], 'Publication moved')
        master = store.load_state(state, 'instruments'); days = sessions(store.load_state(state, 'calendar'))
        issues = audit_daily(prefix, sorted(set(prefix.date)), master, RuleSet.from_yaml('configs/rule_profiles/main_board.yaml'))
        require(not issues.level.eq('block').any(), 'Historical daily audit blocked')
        path = directory / 'historical-daily-audit-v2.csv'; require(not path.exists(), 'Audit exists'); issues.to_csv(path, index=False)
        parts, records = {}, []
        for year in years:
            old = store.load_state(state, 'bars_1d', parts=[year]); add = prefix[prefix.date.map(lambda d: str(d.year) == year)]
            merged = merge_year(old, add); scope = state['tables']['bars_1d'][year].get('history_scope')
            require(scope is not None and scope['policy'] == 'selected_instruments_v1', 'Unknown/full-market prefix scope')
            instruments_, prelisting = annual_scope(scope, merged, master)
            entry = store.write_partition('bars_1d', year, merged)
            entry['history_scope'] = {'policy': 'selected_instruments_v1', 'instruments': instruments_, 'start': str(merged.date.min()),
                'end': str(merged.date.max()), 'full_market': False, 'retained_entry': state['tables']['bars_1d'][year],
                'input_receipt': str(RECEIPT), 'input_receipt_sha256': file_sha(RECEIPT), 'prior_declared_prelisting_names': prelisting}
            parts[year] = entry; records.append({'year': year, 'old_rows': len(old), 'added_rows': len(add), 'rows': len(merged), 'scope_instruments': instruments_, 'prior_declared_prelisting_names': prelisting})
        prospective = {'tables': {**state['tables'], 'bars_1d': {**state['tables']['bars_1d'], **parts}}}
        publication_references(state, prospective, years)
        bid = store.write_batch({'bars_1d': parts}, note='batch19: approved EMA55 verified histories appended; exact old rows and actual annual partial scopes retained')
        audit = store.commit_audit(bid, issues, RuleSet.from_yaml('configs/rule_profiles/main_board.yaml').config_fingerprint(),
            [str(min(prefix.date)), str(max(prefix.date))], scope='selected_history', input_state=prospective)
        store.publish(bid); sid = store.snapshot('batch19 EMA55 verified continuous eight-stock history; annual partial scopes, prelisting absence explicit')
        require(days == sessions(store.load('calendar', sid)), 'Calendar changed')
        result = {'status': 'published', 'batch_id': bid, 'snapshot': sid, 'old_snapshot': SNAPSHOT, 'added_rows': len(prefix), 'years': years,
            'partitions': records, 'audit_id': audit, 'audit_sha256': file_sha(path), 'warnings': issues.groupby('rule').size().to_dict(),
            'input_receipt_sha256': file_sha(RECEIPT), 'all_other_tables_unchanged': True, 'full_market': False, 'strict_financial_usable_rows': 0}
        save(directory / 'publication.json', result)
    archive(root, f'Batch19 EMA history published {bid}, 19560 added rows, prior rows retained')
    return result


def prepare(root, directory):
    custody(directory); require(not (directory / 'prepared-configs.json').exists(), 'Configs exist')
    publication = read(directory / 'publication.json'); store = Store(root); state = store.state(publication['snapshot'])
    days = sessions(store.load_state(state, 'calendar')); bars = store.load_state(state, 'bars_1d', filters=[('instrument', 'in', list(POOL))])
    ready = {}
    for instrument in POOL:
        traded = bars[bars.instrument.eq(instrument) & bars.is_trading].sort_values('date')
        require(len(traded) >= 60, 'EMA history too short'); seeded = traded.iloc[59].date
        ready[instrument] = str(days[days.index(seeded) + 1])
    start_day = max(date.fromisoformat(d) for d in ready.values()); end_day = date.fromisoformat(END)
    actions = store.load_state(state, 'corp_actions', filters=[('instrument', 'in', list(POOL))])
    require(not len(actions[actions.rights_ratio.gt(0) & actions.ex_date.ge(ANCHOR) & actions.ex_date.le(end_day)]), 'Rights action unsupported')
    base = yaml.safe_load(CONFIG.read_text(encoding='utf-8')); records = []
    for label, end in zip(LABELS, (days[days.index(start_day) + 59], end_day), strict=True):
        cfg = StrategyConfig.model_validate({**base, 'name': base['name'] + f' {label}', 'snapshot': publication['snapshot'], 'start': start_day, 'end': end})
        path = directory / f'{label}.yaml'; require(not path.exists(), 'Prepared YAML exists')
        path.write_text(yaml.safe_dump(cfg.model_dump(mode='json'), allow_unicode=True, sort_keys=False), encoding='utf-8')
        records.append({'label': label, 'config_file': str(path), 'config_sha256': file_sha(path), 'start': str(start_day), 'end': str(end),
            'sessions': sum(start_day <= d <= end for d in days), 'snapshot': publication['snapshot']})
    save(directory / 'prepared-configs.json', {'configs': records, 'first_current_prior_ema60': ready,
        'method': 'Next verified calendar session after each 60th traded close; maximum across original eight; no return selection'})
    archive(root, f'Batch19 short/full EMA configs frozen, earliest joint decision {start_day}')
    return records


def bound_config(root, directory, label):
    custody(directory); record = next(r for r in read(directory / 'prepared-configs.json')['configs'] if r['label'] == label)
    require(file_sha(record['config_file']) == record['config_sha256'], 'Prepared config changed')
    cfg = StrategyConfig.model_validate(yaml.safe_load(Path(record['config_file']).read_text(encoding='utf-8'))).model_dump(mode='json')
    approved = StrategyConfig.model_validate(yaml.safe_load(CONFIG.read_text(encoding='utf-8'))).model_dump(mode='json')
    for key in ('name', 'snapshot', 'start', 'end'): approved[key] = cfg[key]
    require(cfg == approved and (cfg['snapshot'], cfg['start'], cfg['end']) == (record['snapshot'], record['start'], record['end']), 'Prepared economic config differs')
    publication = read(directory / 'publication.json'); require(cfg['snapshot'] == publication['snapshot'], 'Config publication differs')
    calendar = sessions(Store(root).load('calendar', cfg['snapshot']))
    require(record['sessions'] == sum(date.fromisoformat(cfg['start']) <= d <= date.fromisoformat(cfg['end']) for d in calendar), 'Session count differs')
    return record


def reference_ema(prices, period):
    result = [math.nan] * len(prices)
    if len(prices) < period: return result
    total = 0.
    for px in prices[:period]: total += px
    value = total / period; result[period - 1] = value; alpha = 2 / (period + 1)
    for k in range(period, len(prices)):
        value = (prices[k] - value) * alpha + value; result[k] = value
    return result


def signal_check(root, output):
    cfg = read(output / 'config.json')['config']; store = Store(root); state = store.state(cfg['snapshot'])
    filters = [('instrument', 'in', list(POOL))]; calendar = sessions(store.load_state(state, 'calendar'))
    view = with_adjusted(store.load_state(state, 'bars_1d', filters=filters), store.load_state(state, 'adj_factors', filters=filters), store.load_state(state, 'adj_coverage', filters=filters))
    references = {}; counts = {}
    for instrument in POOL:
        bars = view[view.instrument.eq(instrument)].sort_values('date'); traded = bars[bars.is_trading]
        table = pd.DataFrame({f'ema{p}': reference_ema(traded.close_adj.tolist(), p) for p in PERIODS}, index=traded.date)
        references[instrument] = table.reindex(calendar).ffill()
        counts[instrument] = {'history_start': str(traded.date.iloc[0]), 'traded_rows': int(traded.date.le(date.fromisoformat(cfg['end'])).sum()),
            'paused_rows': int((bars.date.le(date.fromisoformat(cfg['end'])) & ~bars.is_trading).sum())}
    factors = pd.read_parquet(output / 'factors.parquet'); targets = pd.read_parquet(output / 'targets.parquet').set_index(['decision_date', 'instrument'])
    expected = {}; maximum_error = 0.
    for row in factors.itertuples():
        prior = calendar[calendar.index(row.date) - 1]; current = references[row.instrument].loc[row.date]; previous = references[row.instrument].loc[prior]
        for p in PERIODS:
            for name, value in ((f'ema{p}', current[f'ema{p}']), (f'prior_ema{p}', previous[f'ema{p}'])):
                actual = getattr(row, name); maximum_error = max(maximum_error, abs(value - actual))
                require(math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-10), 'Independent EMA arithmetic differs')
        buy = bool(current.ema2 > current.ema25 and previous.ema2 < previous.ema25 and current.ema25 > current.ema60)
        sell = bool(current.ema2 < current.ema25 and previous.ema2 > previous.ema25 and current.ema25 > current.ema60)
        require(row.buy == buy and row.sell == sell and row.pool_index == POOL.index(row.instrument) and row.prior_date == prior, 'Strict EMA flags/order/date differs')
        target = targets.loc[(row.date, row.instrument)]
        require(target['buy'] == buy and target['sell'] == sell and target.priority == row.pool_index and target.rebalance, 'EMA target differs')
        expected.setdefault(row.date, {})[row.instrument] = (buy, sell)
    require(len(factors) == len(expected) * 8 and all(set(r) == set(POOL) for r in expected.values()), 'EMA factor scope incomplete')
    order_checks = []
    for child in read(output / 'subruns.json')['backtests']:
        folder = Path(child['output']); doc = read(folder / 'config.json')['config']; x = doc['execution']; p = doc['portfolio']
        inputs = build(load(store, state, doc['start'], doc['end'], x['liquidity_window'], doc['execution_instruments']), doc['start'], doc['end'], doc['boards'], x['liquidity_window'], x['liquidity_override'])
        rules = RuleSet.from_yaml(folder / 'rules.yaml'); multiplier = x['fee_multiplier']
        if multiplier != 1: rules = rules.scaled(commission_rate=multiplier, stamp_tax=multiplier, transfer_fee=multiplier, min_commission=multiplier)
        book = Book(doc['initial_cash'], calendar=inputs.calendar); orders = []; max_held = 0
        for k, day in enumerate(inputs.dates):
            quotes = inputs.market[day]; book.start_day(day, inputs.actions.get(day, ()), quotes)
            if k:
                decision = inputs.dates[k - 1]; signals = expected[decision]
                def execute(intent):
                    result = book.execute({**intent, 'decision_date': decision, 'participation': p['participation']}, quotes.get(intent['instrument'], {}), day, rules, x['slippage'])
                    orders.append({**result, 'exec_date': day})
                for instrument, position in list(book.positions.items()):
                    if position.qty and signals[instrument][1]: execute({'instrument': instrument, 'side': 'sell', 'qty': 'all', 'reason': 'slot_exit'})
                for instrument in POOL:
                    if not signals[instrument][0] or (instrument in book.positions and book.positions[instrument].qty): continue
                    free = 5 - sum(position.qty > 0 for position in book.positions.values())
                    if free > 0: execute({'instrument': instrument, 'side': 'buy', 'amount': book.cash / (free * 1.5), 'reason': 'slot_enter'})
            book.close_day(day, quotes); max_held = max(max_held, sum(position.qty > 0 for position in book.positions.values()))
            require(max_held <= 5, 'Exceeded five EMA slots')
        diff, _ = compare_tables({'orders': read(folder / 'orders.json'), 'equity': read(folder / 'equity.json'), 'cash_events': read(folder / 'cash_events.json')},
            canonical({'orders': orders, 'equity': book.equity_rows, 'cash_events': book.cash_events}), 1e-9, 0)
        require(not diff, f'Independent EMA intents/ledger replay differ: {diff[:2]}')
        order_checks.append({'scenario': child['scenario'], 'sessions_checked': len(inputs.dates), 'orders_checked': len(orders), 'max_positions': max_held,
            'partial_orders': sum(o['status'] == 'partial' for o in orders), 'rejected_orders': sum(o['status'] == 'rejected' for o in orders)})
    return {'sessions_checked': len(expected), 'ema_stock_days_checked': len(factors), 'ema_values_checked': len(factors) * 6,
        'maximum_absolute_arithmetic_difference': maximum_error, 'history_counts': counts, 'orders': order_checks,
        'method': 'Independent SMA seed and EMA recurrence/strict comparisons; independent original ordered slot branches using sole Book for cash/fees/NAV'}


def protect(root, directory):
    prefix = incoming_history(root, directory); baseline = read(directory / 'baseline.json'); store = Store(root); current = store.published()
    require(all(file_sha(store.root / p) == sha for p, sha in baseline['controls'].items()), 'Old controls changed')
    require(all((store.root / p).stat().st_size == row['size'] and (store.root / p).stat().st_mtime_ns == row['mtime_ns'] for p, row in baseline['partitions'].items()), 'Old partitions changed')
    publication = read(directory / 'publication.json'); publication_references(baseline['published'], current, publication['years'])
    require(current['batch_id'] == publication['batch_id'] and current['tables'] == store.state(publication['snapshot'])['tables'], 'Published snapshot moved')
    for year in publication['years']:
        before = store.load_state(baseline['published'], 'bars_1d', parts=[year]); after = store.load_state(current, 'bars_1d', parts=[year])
        incoming = prefix[prefix.date.map(lambda d: str(d.year) == year)]
        retain_old_rows(before, after, ['date', 'instrument']); retain_old_rows(incoming, after, ['date', 'instrument'])
        require(len(after) == len(before) + len(incoming), 'Extra/missing published rows')
        scope = current['tables']['bars_1d'][year]['history_scope']; require(scope['instruments'] == sorted(set(after.instrument)), 'Annual scope differs from rows')
    result = {'status': 'ok', 'old_controls_unchanged': len(baseline['controls']), 'old_partition_metadata_unchanged': len(baseline['partitions']),
        'all_prior_rows_retained': True, 'added_rows_verified': len(prefix), 'other_tables_unchanged': True, 'partition_check': 'size/mtime only, not fresh full old partition hashes'}
    save(directory / 'protection.json', result); return result


def checks(root, directory):
    require(not (directory / 'checks.json').exists(), 'Checks exist')
    commands = [[sys.executable, '-m', 'pytest', '-q', 'tests/integration/test_ema_strategy.py', 'tests/unit/test_signal_slots.py',
        'tests/unit/test_history_scope.py', 'tests/unit/test_batch19_gates.py', 'tests/integration/test_rsi_strategy.py', 'tests/integration/test_strategy_scope.py'],
        [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests', 'scripts/prepare_handoff.py', 'scripts/build_strategy_catalog.py',
            'scripts/verify_financial_import.py', 'scripts/run_strategy_batch19.py'], ['git', 'diff', '--check']]
    results = []
    for command in commands:
        process = subprocess.run(command, capture_output=True, text=True); results.append({'command': command, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        print(results[-1], flush=True); require(process.returncode == 0, 'Checks failed')
    require(CORE <= {p.name for p in directory.iterdir() if p.is_file()}, 'Evidence incomplete')
    save(directory / 'checks.json', {'commands': results, 'implementation_sha256': {p: file_sha(p) for p in sorted(FILES)},
        'evidence_sha256': {p.name: file_sha(p) for p in directory.iterdir() if p.is_file()}})
    return {'status': 'ok'}


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch19-verification.json'); require(not output.exists(), 'Receipt exists'); custody(directory)
    checked = read(directory / 'checks.json')
    require(len(checked['commands']) == 3 and all(r['returncode'] == 0 for r in checked['commands']), 'Checks incomplete')
    require(FILES <= set(checked['implementation_sha256']) and CORE <= set(checked['evidence_sha256']), 'Checked sets incomplete')
    for p, sha in checked['implementation_sha256'].items(): require(file_sha(p) == sha, 'Checked code changed')
    for p, sha in checked['evidence_sha256'].items(): require(file_sha(directory / p) == sha, 'Checked evidence changed')
    verified = []
    for label in LABELS:
        record = bound_config(root, directory, label); doc = read(directory / f'{label}-verification.json'); run = read(directory / f'{label}-run.json')
        require(doc['status'] == 'ok' and doc['run'] == run and doc['original_unchanged'], 'Verification differs')
        expected = read(directory / f'{label}-freeze.json')['config']
        require(expected == StrategyConfig.model_validate(yaml.safe_load(Path(record['config_file']).read_text(encoding='utf-8'))).model_dump(mode='json'), 'Freeze differs')
        for result in (run, doc['reproduction']):
            integrity = verify_run(root, result['run_id']); require(integrity['status'] == 'ok', 'Run integrity failed')
            actual = read(Path(integrity['output']) / 'config.json'); require(actual['config'] == expected, 'Run economic config differs')
            require(actual['source']['bytes_sha256'] == custody(directory)['approval']['source_sha256'], 'Run source differs')
        require(doc['reproduction']['reproduction']['result'] == 'match' and doc['reproduction']['reproduction']['differences'] == 0, 'Offline mismatch')
        require(read(Path(run['output']) / 'report.json') == doc['report'] and doc['hand_check']['sessions_checked'] == record['sessions'], 'Report/coverage differs')
        require(doc['hand_check']['ema_stock_days_checked'] == record['sessions'] * 8 and {r['scenario'] for r in doc['hand_check']['orders']} == {'base', 'fees_x2', 'slippage_x2'}, 'Arithmetic/scenario coverage incomplete')
        verified.append(doc)
    fees = read(directory / 'fee-hand-checks.json')
    require(fees['status'] == 'ok' and {(r['run_id'], r['scenario']) for r in fees['checks']} == {(d['run']['run_id'], s) for d in verified for s in ('base', 'fees_x2', 'slippage_x2')} and len(fees['checks']) == 6, 'Fee coverage incomplete')
    old = read(directory / 'old-rsi-reproduction.json')
    require(old['status'] == 'ok' and old['old_original_unchanged'] and old['reproduction']['reproduction']['result'] == 'match' and old['reproduction']['reproduction']['differences'] == 0, 'Old RSI mismatch')
    require(old['original_run_id'] == read('docs/handoff/2026-10-05-batch7-verification.json')['run_id'], 'Old RSI identity differs')
    for rid in (old['original_run_id'], old['reproduction']['run_id']): require(verify_run(root, rid)['status'] == 'ok', 'Old RSI integrity failed')
    protection = protect(root, directory); frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok=False)
    for p, sha in checked['implementation_sha256'].items():
        target = frozen / Path(p).name; require(not target.exists(), 'Frozen filename collision'); shutil.copyfile(p, target); require(file_sha(target) == sha, 'Frozen bytes differ')
    result = {'status': 'ok', 'environment': environment(), 'publication': read(directory / 'publication.json'), 'approval': custody(directory)['approval'],
        'prepared': read(directory / 'prepared-configs.json'), 'strategies': verified, 'fees': fees, 'old_rsi_reproduction': old, 'protection': protection, 'checks': checked,
        'original_strategy_complete': False, 'progress': archive(root, 'Batch19 approved EMA55 three costs, independent arithmetic/slots/fees, offline match and old RSI/asset protection complete'),
        'limitations': ['Original platform equivalence unproved', 'Fixed ex-post source pool before source date', 'Partial selected history, not full market', 'Fee/trading rule assumptions retained; strict financial usable zero']}
    save(output, result); return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['start', 'publish', 'prepare', 'run', 'verify', 'reproduce-old', 'fees', 'checks', 'finish'])
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=Path('data/staging/strategies-batch19/20261005-ema55'))
    parser.add_argument('--label', choices=LABELS, default=LABELS[0]); args = parser.parse_args()
    if args.action == 'run': result = run_one(args.root, args.directory, bound_config(args.root, args.directory, args.label)['config_file'], batch_label='Batch19')
    elif args.action == 'verify':
        bound_config(args.root, args.directory, args.label); result = verify_one(args.root, args.directory, args.label, checker=signal_check, batch_label='Batch19')
    elif args.action == 'fees': result = fee_checks(args.directory)
    else: result = {'start': start, 'publish': publish, 'prepare': prepare, 'reproduce-old': reproduce_old, 'checks': checks, 'finish': finish}[args.action](args.root, args.directory)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
