"""Preserve blocked valuation attempts without presenting their returns as valid."""
import argparse
from collections import Counter
import json
from pathlib import Path

from observe.data.store import Store
from observe.integrity import verify_run
from observe.runs import file_sha, write_json
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch11 import arithmetic
from scripts.run_strategy_batch4 import read, protect, fee_checks
from scripts.verify_strategy_batch3 import require


def child_summary(path):
    status = read(path / 'status.json'); issues = status.get('issues', [])
    groups = {}
    for issue in issues:
        key = (issue['kind'], issue.get('instrument'))
        group = groups.setdefault(key, {'kind': key[0], 'instrument': key[1], 'first': issue, 'occurrences': 0})
        group['occurrences'] += 1
    return {'status': status['status'], 'status_sha256': file_sha(path / 'status.json'),
            'issue_counts': dict(Counter(r['kind'] for r in issues)), 'first_issues': list(groups.values()),
            'blocked': status.get('blocked'), 'returns_usable': status['status'] in ('success', 'success_limited')}


def diagnose(root, directory, run_id):
    out = directory / 'delisting-input-diagnosis'; out.mkdir(exist_ok = False)
    path = Path(root) / 'runs' / run_id
    children = sorted((path / 'variants').glob('*/status.json'))
    completed = [p.parent for p in children if read(p)['status'] == 'blocked']
    require(bool(completed), 'No completed blocked child')
    records = [child_summary(p) for p in completed]
    codes = sorted({g['instrument'] for r in records for g in r['first_issues'] if g['instrument']})
    store = Store(root); config = read(path / 'config.json'); state = store.state(config['snapshot_id'])
    inputs = {}
    for table in ('instruments', 'bars_1d', 'corp_actions'):
        columns = ['date', 'instrument', 'open', 'close', 'preclose', 'is_trading', 'is_st', 'pb_mrq', 'pe_ttm'] if table == 'bars_1d' else None
        frame = store.load_state(state, table, columns = columns, filters = [('instrument', 'in', codes)])
        target = out / f'{table}.parquet'; frame.to_parquet(target, index = False)
        inputs[table] = {'file': str(target), 'sha256': file_sha(target), 'rows': len(frame)}
    result = {'status': 'blocked_data', 'run_id': run_id, 'snapshot': config['snapshot_id'], 'children_at_checkpoint': records,
              'frozen_input_copies': inputs, 'affected_instruments': codes,
              'source_sha256': file_sha(path / 'source.original'), 'config_sha256': file_sha(path / 'config.json'),
              'limits': ['No liquidation or conversion inferred', 'Current master names do not prove historical names',
                         'Blocked child metrics are diagnostic only and not valid strategy returns']}
    write_json(out / 'diagnosis.json', result)
    archive(root, '第十一批BP长区间首项退市持仓阻断已存档；000662于2021-04-13仍持仓，结算证据待补')
    return {'status': 'blocked_data', 'output': str(out / 'diagnosis.json'), 'affected_instruments': codes}


def settlement_evidence(settlement):
    require(settlement.is_file(), 'Archive bounded settlement evidence probes first')
    manifests = []; archived, failed = 0, 0
    for manifest in sorted(settlement.parent.rglob('manifest.json')):
        for response in read(manifest)['responses']:
            if response.get('path'):
                require(file_sha(Path(response['path'])) == response['sha256'], 'Settlement raw response changed')
                archived += 1
            else:
                require(bool(response.get('error')), 'Missing response without recorded failure')
                failed += 1
        manifests.append({'file': str(manifest), 'sha256': file_sha(manifest)})
    return {'manifests': manifests, 'raw_responses_verified': archived, 'failed_attempts_preserved': failed, 'published': False}


def supplemental_checks(directory):
    checks = read(directory / 'supplemental-checks.json')
    required = {__file__, 'tests/unit/test_batch11_archive.py', 'scripts/research_delisting_evidence.py'}
    fingerprints = checks['implementation_sha256']
    normalized = {str(Path(name).resolve()) for name in fingerprints}
    require(all(str(Path(name).resolve()) in normalized for name in required), 'Supplemental checked files missing')
    require(bool(checks['commands']) and all(r['returncode'] == 0 for r in checks['commands']), 'Supplemental checks failed')
    for name, sha in fingerprints.items():
        require(file_sha(Path(name)) == sha, 'Supplemental tested implementation changed')
    return checks


def finish(root, directory):
    output = Path('docs/handoff/2026-10-05-batch11-verification.json')
    if output.exists(): raise FileExistsError(output)
    code = read(directory / 'final-short-verifications/implementation-sha256.json')
    for name, sha in code.items(): require(file_sha(Path(name)) == sha, 'Final short verification implementation changed')
    checks = read(directory / 'checks.json')
    require(all(r['returncode'] == 0 for r in checks['commands']), 'Implementation checks failed')
    for name, sha in checks['implementation_sha256'].items():
        require(file_sha(Path(name)) == sha, 'Tested implementation changed')
    supplemental = supplemental_checks(directory)
    records = []
    for component in ('bp', 'ep'):
        short = directory / 'final-short-verifications' / f'{component}-short-verification.json'
        require(read(short)['reproduction']['reproduction']['result'] == 'match', 'Final short reproduction missing')
        long = read(directory / f'{component}-long-run.json'); path = Path(long['output'])
        require(long['status'] in ('blocked', 'success_limited'), 'Long attempt did not finish')
        require(verify_run(root, path)['status'] == 'ok', 'Blocked experiment integrity failed')
        children = [{'scenario': r['scenario'], **child_summary(Path(r['output']))} for r in long['subruns']['backtests']]
        require(len(children) == 3 and {c['scenario'] for c in children} == {'base', 'fees_x2', 'slippage_x2'}, 'Missing cost scenarios')
        matched = None
        if long['status'] == 'success_limited':
            proof = directory / f'{component}-long-verification.json'; matched = read(proof)
            require(matched['run']['run_id'] == long['run_id'] and matched['status'] == 'ok', 'Long verification identity differs')
            require(matched['reproduction']['reproduction']['result'] == 'match' and matched['reproduction']['reproduction']['differences'] == 0, 'Long reproduction differs')
            require(all(c['returns_usable'] for c in children), 'Successful parent contains blocked child')
            signals = matched['hand_check']
        else:
            require(any(c['status'] == 'blocked' for c in children), 'Blocked parent lacks blocked child')
            signals = arithmetic(root, path)
        records.append({'component': component, 'short_verification': str(short), 'short_verification_sha256': file_sha(short),
                        'long_attempt': long, 'children': children, 'long_returns_usable': matched is not None,
                        'long_signal_hand_check': signals,
                        'long_reproduction': matched['reproduction'] if matched else 'not run while settlement evidence is missing'})
    reviews = directory / 'source-reviews/review.json'
    for record in read(reviews)['sources']:
        require(file_sha(Path(record['source_path'])) == record['source_sha256'] == file_sha(Path(record['source_copy'])), 'Reviewed source changed')
    settlement = Path('data/staging/strategies-batch11/20261005-delisting-evidence/manifest.json')
    settlement_record = settlement_evidence(settlement)
    fees = fee_checks(directory); protection = protect(root, directory)
    frozen = directory / 'implementation-final'; frozen.mkdir(exist_ok = False)
    for name in dict.fromkeys((*code, *supplemental['implementation_sha256'])):
        source = Path(name); relative = source.resolve().relative_to(Path.cwd().resolve())
        target = frozen / relative; target.parent.mkdir(parents = True, exist_ok = True); target.write_bytes(source.read_bytes())
    result = {'status': 'ok_with_blocked_experiments', 'scope': 'Source reviews and valuation optimization accepted; successful long attempts verified, blocked attempts held for settlement evidence',
              'snapshot': '20261005-152153-eb05', 'published_batch': Store(root).published()['batch_id'],
              'experiments': records, 'checks': checks, 'supplemental_checks': supplemental, 'fees': fees, 'protection': protection,
              'source_review': {'file': str(reviews), 'sha256': file_sha(reviews), 'count': len(read(reviews)['sources'])},
              'settlement_evidence': settlement_record,
              'implementation_sha256': {p.relative_to(frozen).as_posix(): file_sha(p) for p in frozen.rglob('*') if p.is_file()},
              'review': {'depth': 'deep', 'scope': 'on target', 'specialists': ['architecture', 'security'], 'remaining_confirmed_findings': []},
              'limitations': ['Delisting settlement and share-conversion evidence incomplete; no last-price liquidation assumption',
                              'Long blocked equity curves/metrics are not valid investment outcomes',
                              'No long blocked replay until dependencies are resolved', 'Main-board component approximation; original source incomplete',
                              'Strict financial usable rows remain zero; 332 prior daily warnings remain', 'Partition protection checks size/mtime only'],
              'progress': archive(root, '第十一批61份累计审查、72项测试/短窗口禁网匹配；BP/EP长区间成功项验收、退市阻断项存档，结算待补')}
    write_json(output, result)
    return {'status': result['status'], 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('action', choices = ['diagnose', 'finish'])
    parser.add_argument('--root', default = 'data'); parser.add_argument('--run-id')
    parser.add_argument('--directory', type = Path, default = Path('data/staging/strategies-batch11/20261005-daily-stock-reviews'))
    args = parser.parse_args()
    if args.action == 'diagnose':
        if not args.run_id: parser.error('--run-id is required')
        result = diagnose(args.root, args.directory, args.run_id)
    else: result = finish(args.root, args.directory)
    print(json.dumps(result, ensure_ascii = False))
