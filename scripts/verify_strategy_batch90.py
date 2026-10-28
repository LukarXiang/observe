"""Accept intraday source research while full-universe/minute/flow trades remain blocked."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pandas as pd

from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import REVIEWS, catalog_strategies
from scripts import diagnose_strategy_batch90 as lifecycle
from scripts import review_strategy_batch90 as stage
from scripts.archive_strategy_progress import archive
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch32 import offline_catalog, validate_catalog
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require
from scripts.verify_strategy_batch89 import network_disabled

DIRECTORY = Path('data/staging/strategies-batch90/20261010-intraday-source-acceptance')
STAGE_RECEIPT = Path('docs/handoff/2026-10-09-batch90-stage-verification.json')
RECEIPT = Path('docs/handoff/2026-10-10-batch90-verification.json')
FILES = ('src/observe/strategy_catalog.py', 'scripts/verify_strategy_batch90.py',
         'scripts/diagnose_strategy_batch90.py', 'tests/unit/test_batch90_lifecycle.py',
         'scripts/review_strategy_batch90.py', 'tests/unit/test_batch90_stage.py')
CORE = {'baseline.json', 'input-binding.json', 'catalog-before.json', 'strategy_catalog.before.py',
        'source-reviews/review.json', 'lifecycle-diagnostics.json', 'reference-replay.json',
        'offline-verification.json', 'offline-catalog.json', 'checked-state.json'}


def source_checks():
    rows = read(stage.DIRECTORY/'sources-frozen.json')
    require(len(rows) == 2, 'Original source scope changed')
    for number, row in enumerate(rows):
        stage.tree(number, stage.DIRECTORY)
        require(row['sha256'] == stage.SOURCE_SHA[number], 'Source SHA changed')
    return rows


def binding(directory=None):
    receipt = read(STAGE_RECEIPT)
    require(receipt['status'] == 'stage_checks_passed' and not receipt['formal_acceptance'] and
            receipt['synthetic_cases'] == 13 and receipt['reference_windows'] == 45763 and
            receipt['component_independent_condition_differences'] == 1 and
            receipt['new_trading_backtests'] == 0, 'Stage scope changed')
    require(all(file_sha(n) == sha for n, sha in receipt['implementation_sha256'].items()), 'Stage code changed')
    for prefix in ('stage_evidence', 'component', 'protection', 'pytest_JUnit'):
        require(file_sha(receipt[prefix+'_file']) == receipt[prefix+'_sha256'], 'Stage receipt binding changed')
    evidence = read(receipt['stage_evidence_file'])
    require(all(file_sha(stage.DIRECTORY/n) == sha for n, sha in evidence['files'].items()), 'Frozen stage evidence changed')
    source_checks()
    result = {'snapshot': stage.SNAPSHOT, 'parent_receipt': str(STAGE_RECEIPT),
        'parent_receipt_sha256': file_sha(STAGE_RECEIPT),
        'parent_stage_evidence_sha256': receipt['stage_evidence_sha256'],
        'not_a_backtest': True, 'new_requests': 0,
        'catalog_binding_policy': 'Original stage replay before two new source reviews; thereafter frozen stage inputs/SHA retained, current catalog independently verified'}
    if directory is not None:
        require(read(directory/'input-binding.json') == result, 'Acceptance inputs changed')
    return result


def preflight(root, directory):
    bound = binding()
    stage.binding(stage.DIRECTORY)
    require(Store(root).published()['batch_id'] == '20261006-145049-5ceb', 'Publication changed')
    checkpoint(root, directory, 'Batch90 source acceptance starts; original historical universe/minute/flow inputs absent')
    save(directory/'input-binding.json', bound)
    shutil.copyfile(Path(root)/'catalog/strategies/latest.json', directory/'catalog-before.json')
    shutil.copyfile('src/observe/strategy_catalog.py', directory/'strategy_catalog.before.py')
    return {'status': 'ok', 'new_trading_backtests': 0}


def audit(root, directory):
    binding(directory)
    result = network_disabled(lifecycle.diagnostics)
    save(directory/'lifecycle-diagnostics.json', {**result, 'socket_HTTP_disabled': True})
    return {'status': 'ok', 'lifecycle_cases': result['case_count'], 'new_trading_backtests': 0}


def replay(root, directory):
    """Replay unchanged stage before its prior catalog code is extended with new reviews."""
    binding(directory)
    old = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    require(file_sha('src/observe/strategy_catalog.py') == file_sha(directory/'strategy_catalog.before.py') and
            read(Path(root)/'catalog/strategies/latest.json')['catalog_id'] == read(directory/'catalog-before.json')['catalog_id'] and
            old.is_file(),
            'Reference stage must replay before catalog code changes')
    require(network_disabled(lambda: stage.diagnostics(stage.DIRECTORY)) == read(stage.DIRECTORY/'diagnostics.json'),
            'Prior synthetic replay differs')
    values, summary = network_disabled(lambda: stage.component(stage.DIRECTORY))
    require(summary == read(stage.DIRECTORY/'component-research.json'), 'Reference summary replay differs')
    pd.testing.assert_frame_equal(values, pd.read_parquet(stage.DIRECTORY/'MA-values.parquet'), check_exact=True)
    save(directory/'reference-replay.json', {'status': 'match', 'differences': 0,
        'reference_windows': len(values), 'prefilter_true': summary['predicate_true'],
        'independent_condition_differences_retained': summary['independent_condition_differences'],
        'prior_synthetic_cases': 13, 'socket_HTTP_disabled': True, 'not_a_backtest': True})
    return read(directory/'reference-replay.json')


def catalog(root, directory):
    binding(directory)
    require(read(directory/'reference-replay.json')['status'] == 'match', 'Prior reference replay missing')
    require(read(directory/'lifecycle-diagnostics.json')['case_count'] == 31, 'Lifecycle audit missing')
    prior = Path(read(directory/'catalog-before.json')['directory'])/'catalog.json'
    old = {r['path']: r for r in read(prior)}
    rows = []
    for name, row in zip(stage.SOURCES, source_checks(), strict=True):
        require(name in REVIEWS and old[name]['review_status'] != '人工审查完成', 'New source review missing')
        rows.append({**row, 'review': REVIEWS[name], 'formal_acceptance': 'source_research_only'})
    now = catalog_strategies(root, 'repo/量化策略源代码')
    records = read(Path(now['output'])/'catalog.json')
    require(len(records) == len(old) == 695 and
            {r['path'] for r in records if r != old[r['path']]} == set(stage.SOURCES), 'Unrelated catalog changed')
    folder = directory/'source-reviews'; folder.mkdir(exist_ok=False)
    save(folder/'review.json', {'sources': rows, 'snapshot': stage.SNAPSHOT, 'catalog': now,
        'previous_catalog_file': str(prior), 'previous_catalog_sha256': file_sha(prior),
        'source_AST_sha256': [hashlib.sha256(ast.dump(stage.tree(i, stage.DIRECTORY)).encode()).hexdigest() for i in range(2)],
        'original_trading_complete': False, 'not_a_backtest': True})
    return archive(root, 'Batch90 two original intraday sources reviewed; final checks pending; no real trades')


def validate_sources(directory):
    binding(directory)
    doc = read(directory/'source-reviews/review.json')
    require(file_sha(doc['previous_catalog_file']) == doc['previous_catalog_sha256'], 'Prior catalog changed')
    for name, row in zip(stage.SOURCES, doc['sources'], strict=True):
        require(row['review'] == REVIEWS[name], 'Source rules changed')


def frozen_operand_binding(directory):
    require(directory == stage.DIRECTORY, 'Unexpected original stage directory')
    binding()
    doc = read(directory/'input-binding.json')
    for key in ('price', 'calendar'):
        require(file_sha(doc[key+'_file']) == file_sha(directory/(key+'-input.parquet')) == doc[key+'_sha256'],
                'Original frozen reference operand changed')
    return doc


def offline(root, directory):
    validate_sources(directory)
    result = network_disabled(lifecycle.diagnostics)
    require({**result, 'socket_HTTP_disabled': True} == read(directory/'lifecycle-diagnostics.json'), 'Lifecycle replay differs')
    require(network_disabled(lambda: stage.diagnostics(stage.DIRECTORY)) == read(stage.DIRECTORY/'diagnostics.json'),
            'Prior original diagnostics differ')
    # Only the stage's prior-catalog binding is adapted; its SHA-pinned arithmetic body and inputs are unchanged.
    with patch.object(stage, 'binding', frozen_operand_binding):
        values, summary = network_disabled(lambda: stage.component(stage.DIRECTORY))
    require(summary == read(stage.DIRECTORY/'component-research.json'), 'Original reference summary differs')
    pd.testing.assert_frame_equal(values, pd.read_parquet(stage.DIRECTORY/'MA-values.parquet'), check_exact=True)
    off = network_disabled(lambda: offline_catalog(root, directory))
    save(directory/'offline-catalog.json', off)
    validate_catalog(root, directory)
    save(directory/'offline-verification.json', {'status': 'match', 'differences': 0,
        'lifecycle_cases': result['case_count'], 'prior_reference_replay': read(directory/'reference-replay.json'),
        'post_catalog_reference_rows_exactly_replayed': len(values),
        'original_component_body_changed': False, 'operand_binding_adapted_for_current_catalog': True,
        'socket_HTTP_disabled': True, 'synthetic_only_for_lifecycle': True,
        'not_a_backtest': True, 'platform_equivalence_proven': False})
    return {'status': 'match', 'differences': 0}


def commands():
    return [[str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests',
             'scripts/verify_strategy_batch90.py', 'scripts/diagnose_strategy_batch90.py', 'scripts/review_strategy_batch90.py'],
            [sys.executable, '-m', 'scripts.verify_offline_tests', '-q', 'tests/unit/test_batch90_lifecycle.py',
             'tests/unit/test_batch90_stage.py', 'tests/unit/test_pair66_history_resume.py',
             'tests/unit/test_research_metadata.py', 'tests/unit/test_catalog_dependencies.py',
             'tests/unit/test_catalog_schedules.py', 'tests/integration/test_strategy_merge.py'],
            ['git', 'diff', '--check']]


def checks(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    require(CORE-{'checked-state.json'} <= {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()},
            'Core evidence missing')
    before = {n: file_sha(n) for n in FILES}; logs = []; folder = directory/'checks'; folder.mkdir(exist_ok=False)
    for i, command in enumerate(commands()):
        path = folder/f'{i}.log'
        with path.open('x') as out:
            result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
        logs.append({'command': command, 'returncode': result.returncode, 'log': str(path), 'sha256': file_sha(path)})
        require(result.returncode == 0, f'Check failed; preserve log: {path}')
    require(before == {n: file_sha(n) for n in FILES}, 'Code changed during checks')
    for name in FILES:
        path = directory/'implementation-checked'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, path)
    save(directory/'checked-state.json', {'status': 'passed', 'commands': logs, 'implementation_sha256': before,
        'evidence_sha256': {p.relative_to(directory).as_posix(): file_sha(p) for p in directory.rglob('*') if p.is_file()}})
    return {'status': 'passed', 'commands': logs}


def finish(root, directory):
    validate_sources(directory); validate_catalog(root, directory)
    checked = read(directory/'checked-state.json'); off = read(directory/'offline-verification.json')
    require(checked['status'] == 'passed' and checked['implementation_sha256'] == {n: file_sha(n) for n in FILES} and
            CORE-{'checked-state.json'} <= checked['evidence_sha256'].keys() and
            [r['command'] for r in checked['commands']] == commands(), 'Frozen check scope changed')
    require(all(r['returncode'] == 0 and file_sha(r['log']) == r['sha256'] for r in checked['commands']), 'Check logs changed')
    require(all(file_sha(directory/n) == sha for n, sha in checked['evidence_sha256'].items()), 'Evidence changed')
    require(all(file_sha(directory/'implementation-checked'/n) == sha for n, sha in checked['implementation_sha256'].items()),
            'Checked copies changed')
    require(off['status'] == 'match' and off['differences'] == 0 and off['socket_HTTP_disabled'] and
            off['lifecycle_cases'] == 31 and off['prior_reference_replay']['reference_windows'] == 45763 and
            off['prior_reference_replay']['independent_condition_differences_retained'] == 1, 'Offline scope changed')
    require(Store(root).published() == read(directory/'baseline.json')['published'], 'Publication changed')
    protected = protect(root, directory)
    progress = archive(root, 'Batch90 original limit-near-high/shortline source reviews accepted;31 lifecycle/13 prior synthetic and45763 reference MA windows offline match0; real signals/trades0')
    save(RECEIPT, {'status': 'ok', 'snapshot': stage.SNAPSHOT, 'input_binding': binding(directory),
        'reviews': read(directory/'source-reviews/review.json'), 'checks': checked,
        'lifecycle_diagnostics': read(directory/'lifecycle-diagnostics.json'),
        'offline_verification': off, 'offline_catalog': read(directory/'offline-catalog.json'),
        'protection': protected, 'progress': progress, 'newly_reviewed': 2,
        'original_signal_rows': 0, 'new_trading_backtests': 0, 'formal_source_research_acceptance': True,
        'original_strategy_complete': False, 'not_a_backtest': True,
        'restore_required': ['Original historical full pool/name/ST and Panel limit/adjustment semantics',
            'Complete index/stock1m prices and money_flow.change_pct historical vintage',
            'MACD platform seeds/requests and quote availability before_open',
            'Platform position mapping/immediate fills and fund-vs-stock fee scope; unique Book intraday execution']})
    save(directory/'final-binding-verification.json', {'status': 'ok', 'receipt_sha256': file_sha(RECEIPT),
        'checked_state_sha256': file_sha(directory/'checked-state.json'), 'catalog_sha256': progress['catalog_sha256'],
        'final_script_sha256': file_sha(__file__)})
    return {'status': 'ok', 'checkpoint': progress['checkpoint'], 'manually_reviewed': progress['manually_reviewed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'audit', 'replay', 'catalog', 'offline', 'checks', 'finish'))
    parser.add_argument('--root', default='data'); parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root, args.directory), ensure_ascii=False, default=str))
