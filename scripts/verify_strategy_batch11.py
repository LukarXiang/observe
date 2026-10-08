"""Archive final-code short reproductions and the valuation batch acceptance."""
import argparse
import json
from pathlib import Path

from observe.runs import environment, file_sha, write_json
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch11 import verify
from scripts.run_strategy_batch4 import read, protect, fee_checks
from scripts.verify_strategy_batch3 import require

CODE = ('src/observe/features.py', 'src/observe/strategies.py', 'src/observe/replay.py', 'src/observe/strategy_catalog.py',
        'scripts/run_strategy_batch11.py', 'scripts/review_strategy_batch11.py', 'scripts/verify_strategy_batch11.py',
        'scripts/research_joinquant_ema.py', 'tests/unit/test_feature_panel.py')


def short_final(root, directory):
    target = directory / 'final-short-verifications'; target.mkdir(exist_ok = False)
    for component in ('bp', 'ep'):
        copied = target / f'{component}-short-run.json'
        copied.write_bytes((directory / copied.name).read_bytes())
        print(json.dumps(verify(root, target, component, short = True), ensure_ascii = False), flush = True)
    write_json(target / 'implementation-sha256.json', {name: file_sha(Path(name)) for name in CODE})
    return {'status': 'ok', 'directory': str(target)}


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch11-verification.json')
    if output.exists(): raise FileExistsError(output)
    for name, sha in read(directory / 'final-short-verifications/implementation-sha256.json').items():
        require(file_sha(Path(name)) == sha, 'Final short verification code changed')
    checks_path = directory / 'checks.json'; checks = read(checks_path)
    require(all(row['returncode'] == 0 for row in checks['commands']), 'Batch checks failed')
    for name, sha in checks['implementation_sha256'].items():
        require(file_sha(Path(name)) == sha, 'Tested implementation changed')
    records = []
    for component in ('bp', 'ep'):
        for period, folder in (('short', directory / 'final-short-verifications'), ('long', directory)):
            proof = folder / f'{component}-{period}-verification.json'; record = read(proof)
            require(record['status'] == 'ok' and record['reproduction']['reproduction']['result'] == 'match', 'Missing matching run')
            records.append({'file': str(proof), 'sha256': file_sha(proof), 'record': record})
    fees = fee_checks(directory); protection = protect(root, directory)
    review_path = directory / 'source-reviews/review.json'; reviews = read(review_path)
    for record in reviews['sources']:
        require(file_sha(Path(record['source_path'])) == record['source_sha256'] == file_sha(Path(record['source_copy'])), 'Reviewed source changed')
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok = False)
    for name in CODE:
        target = frozen / name; target.parent.mkdir(parents = True, exist_ok = True); target.write_bytes(Path(name).read_bytes())
    result = {'status': 'ok', 'scope': 'Eight source reviews and BP/EP component long-period extension; original strategies incomplete',
              'environment': environment(), 'snapshot': '20261005-152153-eb05', 'published_batch': '20261005-152040-e16e',
              'experiments': records, 'fees': fees, 'protection': protection,
              'checks': {'file': str(checks_path), 'sha256': file_sha(checks_path), 'record': checks},
              'source_review': {'file': str(review_path), 'sha256': file_sha(review_path), 'source_count': len(reviews['sources'])},
              'implementation_sha256': {name: file_sha(frozen / name) for name in CODE},
              'review': {'scope': 'on target', 'specialists': ['architecture', 'security'], 'remaining_confirmed_findings': []},
              'progress': archive(root, '第十一批8份审查、BP/EP长区间三成本/独立核算/禁网复现及旧资产保护完成；全库继续'),
              'limitations': ['Main-board component approximation, not historical CSI800 full strategy',
                              'No parameter tuning or independent final holdout', 'Initial partial month follows existing calendar without forced investment',
                              'Strict historical financial usable rows remain zero; 332 prior daily audit warnings remain',
                              'Partition protection is size/mtime, not fresh full hashing', 'GAC12 and EMA55 variant confirmations remain pending']}
    write_json(output, result)
    return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('action', choices = ['short-final', 'finish'])
    parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', type = Path, default = Path('data/staging/strategies-batch11/20261005-daily-stock-reviews'))
    args = parser.parse_args()
    print(json.dumps((short_final if args.action == 'short-final' else finish)(args.root, args.directory), ensure_ascii = False))
