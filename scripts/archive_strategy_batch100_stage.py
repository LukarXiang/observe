"""Archive only observed Batch100 source-stage evidence; never promote pending arithmetic."""
import json
from pathlib import Path

from observe.data.locks import DATA_WRITER, operation_lock
from observe.runs import file_sha
from scripts.archive_strategy_progress import archive
from scripts import research_strategy_batch100 as study
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import read
from scripts.verify_strategy_batch3 import require

RECEIPT = Path('docs/handoff/2026-10-11-batch100-source-stage.json')
REPORT = Path('docs/tasks/2026-10-11-第一百批行业轮动与50ETF-Xgboost-来源与时点阶段记录.md')
GUARD = Path('data/staging/strategies-batch100/20261011-source-stage-validation/checked.json')


def record():
    require(not RECEIPT.exists(), 'Immutable stage receipt exists'); study.validate()
    guard = read(GUARD)
    require(guard['status'] == 'ok_source_stage' and guard['local_passed'] == guard['isolated_passed'] == 11 and
            guard['skips'] == 0, 'Source checks incomplete')
    for name, sha in {**guard['implementation_sha256'], **guard['evidence_sha256']}.items():
        require(file_sha(name) == sha, 'Checked code/evidence differs: '+name)
    diag = read(study.DIRECTORY/'diagnostics-preread.json')
    require(diag['case_count'] == len(diag['cases']) == 32 and diag['actual_json_roundtrip'] and
            diag['synthetic_only'] and diag['new_trading_backtests'] == 0, 'Diagnostic scope differs')
    require(sum(r['error'] is not None for r in diag['cases']) == 5, 'Original exception count differs')
    protection = read(study.DIRECTORY/'protection.json')
    require(protection['status'] == 'ok' and protection['old_controls_unchanged'] == 6614 and
            protection['old_partition_metadata_unchanged'] == 398, 'Old asset protection incomplete')
    evidence = [str(REPORT), str(GUARD), str(study.DIRECTORY/'prepared.json'),
        str(study.DIRECTORY/'dependencies.json'), str(study.DIRECTORY/'diagnostics-preread.json'),
        str(study.DIRECTORY/'baseline.json'), str(study.DIRECTORY/'protection.json'),
        'docs/handoff/batch100-y2022-032-source-review.yaml', 'docs/handoff/batch100-y2022-026-source-review.yaml',
        'data/staging/strategies-batch100/20261011-prepare-search-interruption/interruption.json',
        'scripts/research_strategy_batch100.py', 'tests/unit/test_batch100_research.py',
        'scripts/verify_strategy_batch100_stage.py', 'scripts/archive_strategy_batch100_stage.py']
    with operation_lock('data', DATA_WRITER):
        progress = archive('data', 'Batch100 source/timing phase:2 full sources/32 original diagnostics/11 local+11 isolated;longest400-row prefix running;formal284 unchanged')
        require((progress['manually_reviewed'], progress['locally_backtested_sources'],
                progress['locally_reproduced_sources'], progress['local_strategy_runs']) == (284, 12, 12, 87),
                'Stage changed formal/trading counts')
        result = {'status': 'ok_source_phase', 'formal_source_research_acceptance': False,
            'formal_trading_acceptance': False, 'original_strategy_complete': False,
            'accepted_checkpoint_unchanged': '20261011-032459-0af3', 'progress': progress,
            'source_sha256': list(study.SOURCE_SHA), 'source_lines': [171, 179], 'original_diagnostics': 32,
            'exceptions_preserved': 5, 'validation': {'local': 11, 'isolated': 11, 'skips': 0, 'copied_files': 247},
            'longest_component': {'status': 'in_progress', 'observed_live_tool_session': 37738,
                'driver_action': 'study', 'directory': str(study.DIRECTORY),
                'expected_windows': 4852, 'completed_windows_proven_at_checkpoint': 0,
                'first_expected_decision': '2006-10-17', 'last_expected_decision': '2026-09-29',
                'expected_feature_columns': 1059, 'offline_match': 'not_yet_verified'},
            'protection': protection, 'new_trading_backtests': 0, 'new_strategy_parent_runs': 0,
            'economic_rule_changes': False, 'real_pickle_loaded': False, 'new_requests_responses_publications': 0,
            'evidence_sha256': {n: file_sha(n) for n in evidence},
            'next_start': 'Poll existing live study handle or inspect terminal command receipt; do not restart on missing interim output; then longest independent audit/fresh network-blocked replay/source acceptance',
            'git': 'User managed; no commit or push by agent'}
        save(RECEIPT, result)
    return {'status': result['status'], 'checkpoint': progress['checkpoint'], 'formal_reviewed': 284}


if __name__ == '__main__': print(json.dumps(record(), ensure_ascii=False))
