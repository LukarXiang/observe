"""Verify the conversion fix and preserve existing long-period experiments."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

import requests

from observe.data.sources.baostock import BaoStock
from observe.integrity import verify_run
from observe.replay import reproduce
from observe.runs import environment, file_sha, write_json
from scripts.archive_strategy_progress import archive
from scripts.run_strategy_batch4 import fee_checks, protect, read
from scripts.verify_strategy_batch3 import require, tree_hash

LEGACY = '20261005-163118-44fbd5-strategy-bed8'
TESTS = ('tests/unit/test_conversion.py', 'tests/unit/test_ledger.py', 'tests/unit/test_portfolio_loop.py',
         'tests/unit/test_star_execution.py', 'tests/integration/test_conversion_execution.py',
         'tests/integration/test_rotation_strategy.py', 'tests/integration/test_multi_ma_strategy.py',
         'tests/integration/test_public_run.py')


def legacy(root, directory):
    output = directory / 'legacy-reproduction.json'
    for path in (output, directory / 'legacy-run.json', directory / 'fee-hand-checks.json'):
        if path.exists(): raise FileExistsError(path)
    original = Path(root) / 'runs' / LEGACY; before = tree_hash(original)
    require(verify_run(root, LEGACY)['status'] == 'ok', 'Legacy integrity failed')
    def forbidden(*args, **kwargs): raise AssertionError('Network disabled for reproduction')
    with patch.object(BaoStock, 'session', forbidden), patch.object(requests.sessions.Session, 'request', forbidden), patch.object(socket.socket, 'connect', forbidden):
        result = reproduce(root, LEGACY)
    require(result['reproduction']['result'] == 'match' and result['reproduction']['differences'] == 0, 'Legacy reproduction mismatch')
    require(tree_hash(original) == before, 'Original experiment changed')
    require(verify_run(root, result['run_id'])['status'] == 'ok', 'Reproduction integrity failed')
    write_json(directory / 'legacy-run.json', {'run_id': LEGACY, 'subruns': read(original / 'subruns.json')})
    fees = fee_checks(directory)
    record = {'status': 'ok', 'original_run_id': LEGACY, 'reproduction': result, 'original_tree_sha256': before,
              'original_unchanged': True, 'network_disabled': True, 'fee_checks': fees}
    write_json(output, record)
    archive(root, f'第十批账本修复后旧配对实验禁网复现：{LEGACY}，match / 0差异')
    return record


def verify(root, directory):
    output = Path('docs/handoff/2026-10-05-batch10-verification.json')
    if output.exists(): raise FileExistsError(output)
    checks = []; logs = directory / 'checks-final'; logs.mkdir(exist_ok = False)
    commands = ([sys.executable, '-m', 'pytest', '-q', *TESTS],
                [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests',
                 'scripts/verify_strategy_batch10.py', 'scripts/review_strategy_batch10.py',
                 'scripts/research_etf510310_rules.py', 'scripts/research_sse_etf_rules.py',
                 'scripts/extract_etf510310_evidence.py', 'scripts/research_etf_tax_rules.py'],
                ['git', 'diff', '--check'])
    for name, command in zip(('pytest', 'ruff', 'diff-check'), commands, strict = True):
        log = logs / f'{name}.log'
        if log.exists(): raise FileExistsError(log)
        child = subprocess.run(command, capture_output = True, text = True, check = False)
        log.write_text(child.stdout + child.stderr, encoding = 'utf-8')
        print(child.stdout + child.stderr, flush = True)
        require(child.returncode == 0, f'{name} failed; see {log}')
        checks.append({'command': command, 'returncode': child.returncode, 'log': str(log), 'sha256': file_sha(log)})
    protection = protect(root, directory)
    reviews_path = directory / 'source-reviews-corrected/review.json'
    reviews = read(reviews_path); old = read(directory / 'legacy-reproduction.json')
    etf_manifests = []; response_count = 0
    for manifest in sorted(Path('data/staging/strategies-batch10/ETF证据').glob('*/manifest.json')):
        document = read(manifest)
        for response in document['responses']:
            require(file_sha(Path(response['path'])) == response['sha256'], 'ETF research response changed')
            response_count += 1
        etf_manifests.append({'file': str(manifest), 'sha256': file_sha(manifest), 'responses': len(document['responses'])})
    require(len(etf_manifests) == 6 and response_count == 51, 'ETF evidence archive incomplete')
    etf_report = Path('docs/tasks/2026-10-05-ETF510310原始规则核查.md')
    sources = ['src/observe/ledger/book.py', 'src/observe/execution.py', 'src/observe/strategy_catalog.py',
               'tests/unit/test_conversion.py', 'tests/integration/test_conversion_execution.py',
               'scripts/verify_strategy_batch10.py', 'scripts/review_strategy_batch10.py']
    frozen = directory / 'implementation'; frozen.mkdir(exist_ok = False)
    fingerprints = {}
    for name in sources:
        source = Path(name); target = frozen / name; target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(source.read_bytes()); fingerprints[name] = file_sha(target)
    result = {'status': 'ok', 'scope': 'Conversion ledger fix, source reviews and ETF primary evidence; no new ETF backtest',
              'environment': environment(), 'published_batch': reviews['published_batch'], 'snapshot': reviews['snapshot'],
              'reviewed_sources': reviews['sources'], 'source_review_file': str(reviews_path), 'source_review_sha256': file_sha(reviews_path),
              'checks': checks, 'legacy_reproduction': old, 'protection': protection, 'implementation_sha256': fingerprints,
              'etf_research': {'manifests': etf_manifests, 'response_sha256_matches': response_count, 'differences': 0,
                               'report': str(etf_report), 'report_sha256': file_sha(etf_report), 'data_published': False},
              'review': {'depth': 'deep', 'scope': 'on target', 'specialists': ['security', 'architecture'],
                        'architecture_findings_fixed': ['conversion destination absent from execution inputs', 'two source pool identities corrected in a new review archive'],
                        'security_findings_fixed': ['legacy preflight rejects partial auxiliary archives'], 'remaining_confirmed_findings': []},
              'diagnosis': {'same_code_root_cause': 'old and new aliased; clearing old erased the converted position',
                            'initial_regression': '12 failed, 1 passed before initial fix; tool output only',
                            'numeric_boundary_regression': '4 failed, 16 passed before boundary fix; tool output only',
                            'adapter_regression': '2 failed before destination/combined-event gates; tool output only'},
              'progress': archive(root, '第十批19份审查、Book份额转换修复、旧配对禁网复现及资产保护验收完成；ETF完整执行仍待依赖'),
              'limitations': ['No ETF data publication or ETF strategy backtest', 'Dividend record-date entitlement remains unsupported',
                              'ETF tax primary source, provider adjustment formula, daily reference price and historical state remain unproved',
                              'Current stock candidate gate remains stock-only; no asset-kind workaround',
                              'Historical strict financial usable rows remain zero; 332 existing daily audit warnings remain',
                              'Partition protection checks size/mtime, not a fresh full byte hash',
                              'Conversion atomicity is local to _convert; start_day is not an all-events transaction',
                              'Original strategies remain incomplete; no parameter optimization or final holdout claim']}
    write_json(output, result)
    return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


def clarify(root, directory):
    """Append precise source wording after validation without rewriting its evidence."""
    original = Path('docs/handoff/2026-10-05-batch10-verification.json')
    output = Path('docs/handoff/2026-10-05-batch10-source-clarification.json')
    if output.exists(): raise FileExistsError(output)
    validation = read(original); require(validation['status'] == 'ok', 'Batch validation incomplete')
    for name in ('src/observe/ledger/book.py', 'src/observe/execution.py', 'tests/unit/test_conversion.py',
                 'tests/integration/test_conversion_execution.py'):
        require(file_sha(Path(name)) == validation['implementation_sha256'][name], 'Tested execution code changed')
    previous = read(directory / 'source-reviews-corrected/review.json')
    path = directory / 'source-reviews-final/review.json'; final = read(path)
    old = {r['strategy_id']: r for r in previous['sources']}; changed = []
    require(set(old) == {r['strategy_id'] for r in final['sources']}, 'Reviewed source identities changed')
    for record in final['sources']:
        before = old[record['strategy_id']]
        require(before['source_sha256'] == record['source_sha256'] == file_sha(Path(record['source_path']))
                == file_sha(Path(record['source_copy'])), 'Source bytes changed')
        if record['review'] != before['review']: changed.append(record['source_path'])
    require(changed == ['repo/量化策略源代码/聚宽2025年精选/94别人“拿的住”的股票最赚钱.txt'], 'Unexpected rule clarification')
    frozen = directory / 'source-clarification'; frozen.mkdir(exist_ok = False)
    command = [str(Path(sys.executable).with_name('ruff')), 'check', 'src', 'tests',
               'scripts/verify_strategy_batch10.py', 'scripts/review_strategy_batch10.py']
    child = subprocess.run(command, capture_output = True, text = True, check = False)
    log = frozen / 'ruff.log'; log.write_text(child.stdout + child.stderr, encoding = 'utf-8')
    require(child.returncode == 0, 'Clarification Ruff failed')
    fingerprints = {}
    for name in ('src/observe/strategy_catalog.py', 'scripts/review_strategy_batch10.py', 'scripts/verify_strategy_batch10.py'):
        target = frozen / name; target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(Path(name).read_bytes()); fingerprints[name] = file_sha(target)
    result = {'status': 'ok', 'scope': 'Source metadata clarification only; tested execution and test files unchanged',
              'previous_validation': str(original), 'previous_validation_sha256': file_sha(original),
              'review_file': str(path), 'review_sha256': file_sha(path), 'changed_source_reviews': changed,
              'clarification': 'Daily value is sqrt(sum((r-mean(r))^2*w)/sum(w)); r=close/open-1, w=log(volume); then 20-day std/mean',
              'reviewed_source_bytes_unchanged': len(final['sources']), 'implementation_sha256': fingerprints,
              'ruff': {'command': command, 'returncode': child.returncode, 'log': str(log), 'sha256': file_sha(log)},
              'progress': archive(root, '第十批最终来源公式补充存档：94号权重和开平方明确；此前97项测试、旧实验复现与保护证据保留')}
    write_json(output, result)
    return {'status': 'ok', 'output': str(output), 'progress': result['progress']['output']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__); parser.add_argument('action', choices = ['legacy', 'verify', 'clarify'])
    parser.add_argument('--root', default = 'data')
    parser.add_argument('--directory', default = 'data/staging/strategies-batch10/20261005-ledger-conversion')
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root, Path(args.directory)), ensure_ascii = False))
