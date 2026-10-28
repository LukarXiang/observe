"""Batch95 acceptance preparation: immutable ancestry, live inventory and fresh replay.

This preparatory driver cannot publish a catalog or accept trading/source research.
The later acceptance must bind these checked inputs and the final review surface.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.data.store import Store
from observe.runs import file_sha
from observe.strategy_catalog import read_source
from scripts import review_strategy_batch95 as stage
from scripts.archive_strategy_progress import archive
from scripts.recover_strategy_batch94_reference import network_blocked
from scripts.review_strategy_batch18 import save
from scripts.review_strategy_batch59 import index_operand
from scripts.run_strategy_batch4 import checkpoint, protect, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch95/20261010-magic3-v2-acceptance-preparation-v3')
RECEIPT = Path('docs/handoff/2026-10-10-batch95-acceptance-preparation.json')
PARENT = Path('docs/handoff/2026-10-10-batch95-numerical-stage.json')
PARENT_SHA = '6fccfda65518065fd57ef4eb5cfe1ca6a34cb3bd4fdba514408967cb090e4b44'
FILES = ('scripts/verify_strategy_batch95.py', 'tests/unit/test_batch95_preparation.py', *stage.FILES,
    'scripts/review_strategy_batch94.py', 'scripts/recover_strategy_batch94_reference.py',
    'scripts/review_strategy_batch59.py', 'scripts/review_strategy_batch73.py',
    'scripts/archive_strategy_progress.py', 'scripts/review_strategy_batch18.py',
    'scripts/run_strategy_batch4.py', 'scripts/verify_strategy_batch3.py',
    'src/observe/data/store.py', 'src/observe/data/locks.py', 'src/observe/runs.py',
    'src/observe/strategy_catalog.py')
BASELINE_FILES = {'baseline.json', 'progress-start.json', 'implementation-evidence.before.json',
    'historical-evidence.json', 'input-inventory.json', 'implementation-binding.json'}
POOL = ('510050.SH', '510300.SH', '159949.SZ', '159928.SZ')
TABLES = ('instruments', 'bars_1d', 'corp_actions', 'adj_factors', 'adj_coverage', 'bars_5m')
STAGE_REVIEWS = {
    'data/staging/strategies-batch95/20261010-source-independent-review/security/review.json':
        'ed1a481d5f2605b72bd610d1aa3b26448ad35e6487db675852ef4363cf875cdd',
    'data/staging/strategies-batch95/20261010-source-independent-review/architecture/review.json':
        'cd012b37a1b5ad3e2582170479baa45fbd4a9a5c03de7f2d4aa5a7fd983aa6df',
    'data/staging/strategies-batch95/20261010-source-independent-review/assumptions/review-receipt.json':
        '9631a8738c38172371e6454062d86ea6793988155f066d8cad8d1739f985d9d4',
}


def regular(path):
    path = Path(path)
    require(path.is_file() and path.resolve().is_relative_to(Path.cwd().resolve()) and
        not any(p.is_symlink() for p in (path, *path.parents)), 'Unrooted or symlink evidence: '+str(path))
    return path


def require_root(root):
    path = Path(root)
    require(path.resolve() == Path('data').resolve() and path.is_dir() and
        not any(p.is_symlink() for p in (path, *path.parents)), 'Only canonical data root is supported')


def bound(path, expected):
    content = regular(path).read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected, 'Bound bytes changed: '+str(path))
    return content


def implementation(directory=None, manifest=None):
    hashes = {n: file_sha(regular(n)) for n in FILES}
    if manifest is not None:
        require(isinstance(manifest, dict) and set(manifest) == set(FILES) and manifest == hashes,
            'Implementation inventory differs')
        for name, sha in manifest.items(): bound(directory/'implementation'/name, sha)
    return hashes


def ancestry():
    """Live catalog may advance; its old identity is checked using a frozen copy."""
    receipt = json.loads(bound(PARENT, PARENT_SHA)); evidence = {str(PARENT): PARENT_SHA}
    require(receipt['status'] == 'ok' and not receipt['formal_source_research_acceptance'] and
        not receipt['formal_trading_acceptance'] and receipt['new_trading_backtests'] == 0,
        'Historical stage scope differs')
    for name, sha in receipt['evidence_sha256'].items(): bound(name, sha); evidence[name] = sha
    bound(receipt['formal_report'], receipt['report_sha256'])
    evidence[receipt['formal_report']] = receipt['report_sha256']
    require(set(receipt['implementation_sha256']) == set(stage.FILES), 'Historical code inventory differs')
    for name, sha in receipt['implementation_sha256'].items():
        frozen = stage.DIRECTORY/'implementation'/name
        bound(name, sha); bound(frozen, sha); evidence[name] = evidence[str(frozen)] = sha
    parent_inventory = {}
    for name, sha in zip(stage.PARENTS, stage.PARENT_SHA, strict=True):
        doc = json.loads(bound(name, sha)); parent_inventory[name] = sha
        require(doc['status'] == 'ok' and doc['manually_reviewed'] == 273, 'Historical acceptance scope differs')
        for key in ('receipt', 'final_binding', 'formal_report'):
            path = doc[key] if key == 'formal_report' else doc[key+'_file']
            bound(path, doc[key+'_sha256']); parent_inventory[path] = doc[key+'_sha256']
    parent = read(stage.PARENTS[0])
    for name, sha in parent['reviewed_implementation_sha256'].items():
        historical = Path(parent['accepted_directory'])/'implementation-checked'/name
        bound(historical, sha); parent_inventory[name] = sha
    require(parent_inventory == read(stage.DIRECTORY/'parent-binding.json'), 'Historical parent inventory differs')
    evidence.update({str(Path(parent['accepted_directory'])/'implementation-checked'/n): sha
        for n, sha in parent['reviewed_implementation_sha256'].items()})
    evidence.update({n: sha for n, sha in parent_inventory.items() if n not in parent['reviewed_implementation_sha256']})
    binding = read(stage.DIRECTORY/'index-binding.json')
    bound(stage.DIRECTORY/'index-input.parquet', binding['frozen_sha256'])
    evidence[str(stage.DIRECTORY/'index-input.parquet')] = binding['frozen_sha256']
    outputs = read(stage.DIRECTORY/'output-binding.json')
    require(set(outputs) == set(stage.OUTPUTS), 'Historical output inventory differs')
    for n, sha in outputs.items(): bound(stage.DIRECTORY/n, sha); evidence[str(stage.DIRECTORY/n)] = sha
    rows = read(stage.DIRECTORY/'source-reviews.json')['sources']
    require(len(rows) == 3, 'Source inventory differs')
    for number, (name, sha) in enumerate(zip(stage.SOURCES, stage.SOURCE_SHA, strict=True)):
        original = Path('repo/量化策略源代码')/name; frozen = stage.DIRECTORY/'sources'/f'{number}.source'
        bound(original, sha); bound(frozen, sha); evidence[str(original)] = evidence[str(frozen)] = sha
        text = read_source(frozen)[0]; functions = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
        require(rows[number]['path'] == name and rows[number]['sha256'] == sha and
            rows[number]['lines'] == len(text.splitlines()) == (249 if number == 2 else 171) and
            rows[number]['functions'] == [n.name for n in functions] and
            len(functions) == (15 if number == 2 else 13) and rows[number]['pool_order'] == list(stage.POOL[number]) and
            rows[number]['duplicate_of'] == (0 if number == 1 else None) and
            rows[number]['minimum_commission'] == (0 if number == 2 else 5), 'Full original source review differs')
    for name, sha in STAGE_REVIEWS.items():
        doc = json.loads(bound(name, sha)); require(not doc['findings'], 'Stage review has findings')
        evidence[name] = sha
        for path, digest in doc.get('evidence', {}).items(): bound(path, digest); evidence[path] = digest
    return evidence


def input_inventory(root):
    require_root(root); published = Store(root).published()
    require(published['batch_id'] == '20261006-145049-5ceb', 'Published input differs')
    tables = {}; paths = {}; connection = duckdb.connect()
    try:
        for table in TABLES:
            entries = published['tables'].get(table, {}); files = [str(regular(Path(root)/r['file'])) for r in entries.values()]
            counts = dict.fromkeys(POOL, 0)
            if files:
                rows = connection.execute('SELECT instrument, count(*) FROM read_parquet(?, union_by_name=true) '
                    'WHERE instrument IN (?, ?, ?, ?) GROUP BY instrument', [files, *POOL]).fetchall()
                counts.update({symbol: int(count) for symbol, count in rows})
            tables[table] = {'published_rows_manifest': sum(r['rows'] for r in entries.values()),
                'observed_original_pool_rows': counts}
            for filename in files:
                stat = Path(filename).stat(); paths[filename] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    finally: connection.close()
    frame, operand = index_operand(root); frozen = read(stage.DIRECTORY/'index-binding.json')
    require(operand == {k: v for k, v in frozen.items() if k != 'frozen_sha256'}, 'Frozen index provenance differs')
    pd.testing.assert_frame_equal(frame, pd.read_parquet(stage.DIRECTORY/'index-input.parquet'), check_exact=True)
    candidates = read(stage.DIRECTORY/'candidate-inputs.json'); candidate_hashes = {}
    for row in candidates['funds'].values():
        bound(row['file'], row['sha256']); candidate_hashes[row['file']] = row['sha256']
        for wire in row.get('wire_responses', []):
            bound(wire['file'], wire['sha256']); candidate_hashes[wire['file']] = wire['sha256']
        stage.prior.native_frame(row)
    require(candidates['missing_from_known_candidates'] == ['159928.XSHE'] and len(candidates['funds']) == 3,
        'Known candidate pool differs')
    return {'published': published, 'tables': tables, 'partition_controls': paths,
        'partition_check': 'size/mtime only; not a fresh full partition hash', 'index': operand,
        'candidate_byte_sha256': candidate_hashes, 'candidate_profiles': candidates['profiles'],
        'missing_native_pool': ['159928.XSHE'], 'original_platform_equivalence_proven': False,
        'fund_60m_admitted': False, 'new_requests': 0, 'new_provider_rows': 0, 'new_trading_backtests': 0}


def prepare(root, directory):
    require_root(root); require(directory == DIRECTORY and not directory.exists(), 'Preparation directory exists or differs')
    historical = ancestry(); inputs = input_inventory(root); hashes = implementation()
    with operation_lock(root, DATA_WRITER):
        require(historical == ancestry() and inputs['published'] == Store(root).published(), 'Inputs changed during preflight')
        checkpoint(root, directory, 'Batch95 acceptance preparation starts; formal273/trade12/parents77 unchanged')
        save(directory/'historical-evidence.json', historical); save(directory/'input-inventory.json', inputs)
        for name, sha in hashes.items():
            target = directory/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bound(name, sha))
        save(directory/'implementation-binding.json', hashes)
        save(directory/'preparation-binding.json', {n: file_sha(directory/n) for n in sorted(BASELINE_FILES)})
    return {'status': 'prepared', 'formal_source_research_acceptance': False, 'new_trading_backtests': 0}


def validate(directory):
    require(directory == DIRECTORY, 'Preparation directory differs')
    binding = read(directory/'preparation-binding.json')
    require(set(binding) == BASELINE_FILES, 'Preparation evidence inventory differs')
    for name, sha in binding.items(): bound(directory/name, sha)
    implementation(directory, read(directory/'implementation-binding.json'))
    require(ancestry() == read(directory/'historical-evidence.json'), 'Historical evidence changed')
    return read(directory/'implementation-binding.json')


def worker(directory):
    require(sys.dont_write_bytecode and os.environ.get('OBSERVE_BATCH95_WORKER') == '1', 'Fresh worker required')
    hashes = validate(directory); target = directory/'replay'
    require(not target.exists(), 'Replay output already exists'); target.mkdir()
    with network_blocked(): outputs = stage.compute(stage.DIRECTORY)
    require(set(outputs) == set(stage.OUTPUTS), 'Replay output inventory differs')
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, pd.read_parquet(stage.DIRECTORY/name), check_exact=True)
            with (target/name).open('xb') as stream: value.to_parquet(stream, index=False)
        else:
            require(value == read(stage.DIRECTORY/name), 'Replay JSON differs: '+name); save(target/name, value)
    scope = (len(outputs['rsrs-magic3.parquet']), len(outputs['rsrs-v2-reference.parquet']),
        len(outputs['native-momentum-reference.parquet']), outputs['diagnostics.json']['count'])
    require(scope == (4662, 4662, 22044, 108), 'Replay numerical scope differs')
    require(hashes == validate(directory), 'Evidence or implementation changed during replay')
    result = {'status': 'match', 'differences': 0, 'scope': list(scope), 'network_forbidden': True,
        'implementation_sha256': hashes, 'file_sha256': {n: file_sha(target/n) for n in stage.OUTPUTS},
        'formal_source_research_acceptance': False, 'formal_trading_acceptance': False, 'new_trading_backtests': 0}
    save(directory/'replay.json', result); return result


def environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONPATH'): env.pop(name, None)
    return env


def replay(directory):
    validate(directory); require(not (directory/'replay-command.json').exists(), 'Replay command already archived')
    command = [sys.executable, '-B', '-m', 'scripts.verify_strategy_batch95', 'worker']
    env = environment(); env['OBSERVE_BATCH95_WORKER'] = '1'; log = directory/'replay.log'
    with log.open('x') as stream: result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    save(directory/'replay-command.json', {'command': command, 'returncode': result.returncode,
        'log': str(log), 'sha256': file_sha(log)})
    require(result.returncode == 0, 'Fresh replay failed; preserve log'); return validate_replay(directory)


def validate_replay(directory):
    result = read(directory/'replay.json'); command = read(directory/'replay-command.json')
    require(command['command'] == [sys.executable, '-B', '-m', 'scripts.verify_strategy_batch95', 'worker'] and
        command['returncode'] == 0 and command['log'] == str(directory/'replay.log') and
        file_sha(regular(command['log'])) == command['sha256'], 'Replay command differs')
    require(result['status'] == 'match' and result['differences'] == result['new_trading_backtests'] == 0 and
        result['scope'] == [4662, 4662, 22044, 108] and result['network_forbidden'] and
        not result['formal_source_research_acceptance'] and not result['formal_trading_acceptance'] and
        result['implementation_sha256'] == validate(directory) and set(result['file_sha256']) == set(stage.OUTPUTS),
        'Replay proof scope differs')
    for name, sha in result['file_sha256'].items(): bound(directory/'replay'/name, sha)
    # Hash agreement identifies bytes. Parsing/comparison proves their numerical meaning.
    counts = []
    for name in stage.OUTPUTS:
        path = regular(directory/'replay'/name); original = regular(stage.DIRECTORY/name)
        if name.endswith('.parquet'):
            frame = pd.read_parquet(path)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(original), check_exact=True)
            counts.append(len(frame))
        else:
            value = read(path); require(value == read(original), 'Replay JSON values differ')
            if name == 'diagnostics.json': cases = value['count']
            else:
                require(value['new_trading_backtests'] == 0 and not value['formal_source_research_acceptance'] and
                    not value['original_strategy_complete'], 'Replay JSON became trading/formal acceptance')
    require(counts == [4662, 4662, 22044] and cases == 108, 'Actual replay row/case counts differ')
    return result


def complete_preparation(root, directory):
    """Record checked preparation only. Later formal acceptance has separate gates."""
    require_root(root); require(not RECEIPT.exists(), 'Preparation receipt exists')
    hashes = validate(directory); result = validate_replay(directory)
    require(input_inventory(root) == read(directory/'input-inventory.json'), 'Live inventory changed')
    with operation_lock(root, DATA_WRITER):
        require(hashes == validate(directory), 'Preparation code changed')
        require(input_inventory(root) == read(directory/'input-inventory.json'), 'Inventory changed while acquiring writer lock')
        require(validate_replay(directory) == result, 'Replay changed while acquiring writer lock')
        protection = protect(root, directory)
        progress = archive(root, 'Batch95 SHA-bound ancestry/live pool inventory/fresh offline replay prepared; formal acceptance pending')
        require((progress['manually_reviewed'], progress['locally_backtested_sources'], progress['local_strategy_runs']) ==
            (273, 12, 77), 'Preparation changed formal totals')
        require(validate_replay(directory) == result, 'Replay changed while archiving preparation')
        save(directory/'progress-preparation.json', progress)
        receipt = {'status': 'ok', 'scope': 'acceptance_preparation_only', 'progress': progress,
            'stage_receipt': str(PARENT), 'stage_receipt_sha256': PARENT_SHA,
            'fresh_offline_replay': result, 'implementation_sha256': hashes, 'protection': protection,
            'formal_source_research_acceptance': False, 'formal_trading_acceptance': False,
            'new_trading_backtests': 0, 'new_requests': 0, 'library_goal_complete': False,
            'evidence_sha256': {str(p): file_sha(p) for p in sorted(directory.rglob('*')) if p.is_file()},
            'next_step': 'New formal catalog candidate plus exact test/isolation and six-role final review; fund/60m admission before trades'}
        save(RECEIPT, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'replay', 'worker', 'checkpoint'))
    parser.add_argument('--root', default='data'); args = parser.parse_args(); require_root(args.root)
    if args.action == 'prepare': result = prepare(args.root, DIRECTORY)
    elif args.action == 'replay': result = replay(DIRECTORY)
    elif args.action == 'worker': result = worker(DIRECTORY)
    else: result = complete_preparation(args.root, DIRECTORY)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
