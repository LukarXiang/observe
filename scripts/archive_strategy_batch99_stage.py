"""Read back Batch99 evidence and checkpoint progress without source acceptance."""
import json
from pathlib import Path

from observe.data.locks import DATA_WRITER, operation_lock
from observe.runs import file_sha
from scripts.archive_strategy_progress import archive
from scripts import research_strategy_batch99 as stage
from scripts import recover_strategy_batch99_namespace as recovery
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

RECEIPT = Path('docs/handoff/2026-10-11-batch99-arithmetic-stage.json')
GUARD = Path('data/staging/strategies-batch99/20261011-stage-validation-recovery/checked.json')
REPORT = Path('docs/tasks/2026-10-11-第九十九批中国VIX与期权突破-算式与输入阶段记录.md')


def record():
    require(not RECEIPT.exists(), 'Immutable stage receipt exists')
    offline = read(recovery.DIRECTORY/'offline-verification.json')
    require(offline['status'] == 'match' and offline['differences'] == 0 and
            offline['fresh_process'] is True and offline['network_blocked'] is True,
            'Fresh offline recovery missing')
    for name, sha in offline['output_sha256'].items():
        require(file_sha(recovery.DIRECTORY/'study'/name) ==
                file_sha(recovery.DIRECTORY/'replay'/name) == sha, 'Persisted replay differs')
    guard = read(GUARD)
    require(guard['status'] == 'ok' and guard['local_passed'] == 44 and
            guard['isolated_passed'] == 21 and guard['skips'] == 0, 'Current validation scope differs')
    for name, sha in {**guard['implementation_sha256'], **guard['evidence_sha256']}.items():
        require(file_sha(name) == sha, 'Checked implementation/evidence differs: '+name)
    old = read(stage.DIRECTORY/'protection.json')
    require(old['status'] == 'ok' and old['old_controls_unchanged'] == 6614 and
            old['old_partition_metadata_unchanged'] == 398, 'Old asset protection missing')
    diag = read(recovery.DIRECTORY/'study/diagnostics.json')
    summary = read(recovery.DIRECTORY/'study/summary.json')
    require(diag['case_count'] == len(diag['cases']) == 44 and summary['rows'] == 5249 and
            summary['rounding_difference_days'] == {'scalar': 319, 'decimal': 151},
            'Persisted diagnostic/reference scope differs')
    specs = ('docs/handoff/batch99-y2021-064-source-review.yaml',
             'docs/handoff/batch99-y2022-018-source-review.yaml')
    evidence = [str(stage.DIRECTORY/'prepared.json'), str(stage.DIRECTORY/'dependencies.json'),
        str(stage.DIRECTORY/'baseline.json'), str(stage.DIRECTORY/'protection.json'),
        *(str(recovery.DIRECTORY/'study'/n) for n in stage.OUTPUTS),
        str(recovery.DIRECTORY/'prepared.json'), str(recovery.DIRECTORY/'offline-verification.json'),
        str(recovery.DIRECTORY/'study-command/command.json'), str(recovery.DIRECTORY/'replay-command/command.json'),
        'data/staging/strategies-batch99/20261011-coverage-reference/coverage.json',
        'data/staging/strategies-batch99/20261011-coverage-reference/rounding-audit.json',
        'data/staging/strategies-batch99/20261011-bounded-option-probes/summary.json',
        'data/staging/strategies-batch99/20261011-test-date-index-failure/failure.json',
        'data/staging/strategies-batch99/20261011-fresh-namespace-recovery/fixture-failure.json',
        str(GUARD), *specs, str(REPORT)]
    with operation_lock('data', DATA_WRITER):
        progress = archive('data', 'Batch99 arithmetic phase:44 original diagnostics/5249 ETF bands/recovery offline match0;formal282 unchanged;option trades blocked')
        require(progress['manually_reviewed'] == 282 and progress['local_strategy_runs'] == 87 and
                progress['locally_backtested_sources'] == progress['locally_reproduced_sources'] == 12,
                'Stage changed formal source/trading counts')
        receipt = {'schema': 1, 'status': 'ok_phase', 'formal_strategy_acceptance': False,
            'scope': 'Full source reading, original arithmetic and input verification; not trading acceptance',
            'progress': progress, 'accepted_checkpoint_unchanged': '20261011-024709-1cdb',
            'sources': [{'path': n, 'sha256': s} for n, s in zip(stage.SOURCE_PATHS, stage.SOURCE_SHA)],
            'summary': summary, 'synthetic_diagnostics': 44, 'offline': offline,
            'validation': {k: guard[k] for k in ('local_passed', 'isolated_passed', 'skips', 'copied_files', 'scope')},
            'protection': old, 'new_trading_backtests': 0, 'original_strategy_complete': False,
            'economic_rule_changes': False, 'source_credentials_executed': False,
            'failed_historical_queries': 2, 'new_responses_rows_publications': 0,
            'evidence_sha256': {n: file_sha(n) for n in evidence},
            'archive_driver_sha256': file_sha(__file__),
            'next_start': 'Formal acceptance still pending: independently read rules/exceptions/rounding and immutable failure/recovery bindings, then only two-source catalog delta; recover option minute/SHIBOR/margin inputs before Book trading',
            'git': 'User managed; no commit or push by agent'}
        save(RECEIPT, receipt)
    return {'status': 'ok_phase', 'checkpoint': progress['checkpoint'], 'formal_reviewed': 282,
            'source_rules_read': 2, 'reference_rows': 5249, 'new_trading_backtests': 0}


if __name__ == '__main__': print(json.dumps(record(), ensure_ascii=False))
