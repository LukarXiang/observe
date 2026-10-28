"""Resume all original 400-row windows in immutable chunks; never repair features."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from filelock import FileLock
import pandas as pd

from observe.data.locks import DATA_WRITER, operation_lock
from observe.runs import file_sha
from scripts import research_strategy_batch100 as study
from scripts.review_strategy_batch18 import save
from scripts.run_strategy_batch4 import checkpoint, read
from scripts.verify_strategy_batch3 import require

DIRECTORY = Path('data/staging/strategies-batch100/20261011-feature-chunk-recovery')
FILES = ('scripts/recover_strategy_batch100.py', 'tests/unit/test_batch100_recovery.py')
CHUNK_SIZE = 100


def ranges(windows, size=CHUNK_SIZE):
    require(windows > 0 and size > 0, 'Invalid chunk scope')
    return [(start, min(start+size, windows)) for start in range(0, windows, size)]


def prepare():
    study.validate()
    failures = read(DIRECTORY.parent/'20261011-driver-timeout-evidence/failures.json')
    require(all(failures['sessions'][m]['terminal'] and failures['sessions'][m]['exit_code'] == 1
                for m in ('study', 'replay')), 'Original drivers not confirmed terminal')
    require(not DIRECTORY.exists(), 'Immutable recovery exists')
    with operation_lock('data', DATA_WRITER):
        checkpoint('data', DIRECTORY, 'Batch100 original drivers terminal timeout; full4852-window chunk recovery started;formal284/trade12/parents87 unchanged')
        for name in FILES:
            target = DIRECTORY/'implementation'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
        operand = pd.read_parquet(study.DIRECTORY/'ETF-input.parquet')
        save(DIRECTORY/'prepared.json', {'original_directory': str(study.DIRECTORY),
            'original_prepared_sha256': file_sha(study.DIRECTORY/'prepared.json'),
            'operand_sha256': file_sha(study.DIRECTORY/'ETF-input.parquet'),
            'original_failure_sha256': file_sha(DIRECTORY.parent/'20261011-driver-timeout-evidence/failures.json'),
            'implementation_sha256': {n: file_sha(n) for n in FILES},
            'windows': len(operand)-400, 'ranges': ranges(len(operand)-400),
            'seed_policy': 'Unchanged original components(segment): each decision recomputes all 400 rows',
            'new_trading_backtests': 0, 'economic_rule_changes': False})
    return {'status': 'prepared', 'chunks': 49, 'windows': len(operand)-400}


def validate():
    study.validate(); doc = read(DIRECTORY/'prepared.json')
    require(doc['original_directory'] == str(study.DIRECTORY) and
            doc['original_prepared_sha256'] == file_sha(study.DIRECTORY/'prepared.json') and
            doc['operand_sha256'] == file_sha(study.DIRECTORY/'ETF-input.parquet'), 'Recovery input changed')
    require(doc['windows'] == 4852 and doc['ranges'] == [list(r) for r in ranges(4852)] and
            doc['economic_rule_changes'] is False, 'Recovery coverage/rules changed')
    for name, sha in doc['implementation_sha256'].items():
        require(file_sha(name) == file_sha(DIRECTORY/'implementation'/name) == sha, 'Recovery implementation changed')
    return doc


def calculate(frame, start, end):
    require(0 <= start < end <= len(frame)-400, 'Chunk outside input range')
    # Calls the unchanged frozen algorithm, including all rolling rows and fresh TA seeds.
    output, summary = study.components(frame.iloc[start:end+400].reset_index(drop=True))
    require(output.decision_date.tolist() == frame.date.iloc[start+400:end+400].tolist(), 'Chunk decision dates differ')
    return output, summary


def check_chunk(folder, start, end, operand_sha):
    doc = read(folder/'completed.json')
    require(doc['range'] == [start, end] and doc['operand_sha256'] == operand_sha and
            doc['network_blocked'] is True and doc['status'] == 'ok', 'Chunk identity differs')
    for name, sha in doc['sha256'].items():
        require(file_sha(folder/name) == sha, 'Chunk artifact changed: '+name)
    frame = pd.read_parquet(folder/'feature-prefix.parquet'); summary = read(folder/'summary.json')
    require(len(frame) == summary['windows'] == end-start and len(frame.columns)-3 == 1059 and
            summary['independent_scalar_cells'] == (end-start)*1059, 'Chunk comparison scope differs')
    return frame, summary


def worker(mode, start, end, folder):
    doc = validate(); require(mode in ('study', 'replay') and [start, end] in doc['ranges'], 'Unknown chunk')
    require(folder.is_relative_to(DIRECTORY/mode/'chunks') and folder.is_dir() and
            not (folder/'completed.json').exists(), 'Worker attempt invalid')
    with study.network_blocked():
        output, summary = calculate(pd.read_parquet(study.DIRECTORY/'ETF-input.parquet'), start, end)
        output.to_parquet(folder/'feature-prefix.parquet', index=False); save(folder/'summary.json', summary)
    result = {'status': 'ok', 'range': [start, end], 'mode': mode,
        'operand_sha256': doc['operand_sha256'], 'network_blocked': True,
        'sha256': {n: file_sha(folder/n) for n in ('feature-prefix.parquet', 'summary.json')}}
    save(folder/'completed.json', result)
    return result


def run_process(command, folder, timeout=600):
    """Stream output immediately and always record terminal timeout/nonzero status."""
    started = time.time(); save(folder/'started.json', {'command': command, 'started_at_unix': started})
    with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        timed_out = False
        try: code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True; process.kill(); code = process.wait()
    result = {'command': command, 'returncode': code, 'timed_out': timed_out,
        'elapsed_seconds': time.time()-started,
        'stdout_sha256': file_sha(folder/'stdout.log'), 'stderr_sha256': file_sha(folder/'stderr.log')}
    save(folder/'command.json', result)
    require(code == 0 and not timed_out, 'Chunk failed; streamed logs and terminal receipt retained')
    return result


def existing(mode, start, end, operand_sha):
    root = DIRECTORY/mode/'chunks'/f'{start:06d}-{end:06d}'
    ready = sorted(root.glob('*/completed.json'))
    require(len(ready) <= 1, 'Multiple completed attempts for chunk')
    if not ready: return None
    folder = ready[0].parent
    check_chunk(folder, start, end, operand_sha)
    command = read(folder/'command.json')
    require(command['returncode'] == 0 and not command['timed_out'], 'Completed chunk lacks successful driver')
    for stream in ('stdout', 'stderr'):
        require(file_sha(folder/(stream+'.log')) == command[stream+'_sha256'], 'Chunk stream changed')
    return folder


def run_chunk(mode, start, end, operand_sha):
    ready = existing(mode, start, end, operand_sha)
    if ready is not None: return {'range': [start, end], 'reused_verified_chunk': True}
    folder = DIRECTORY/mode/'chunks'/f'{start:06d}-{end:06d}'/uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-m', 'scripts.recover_strategy_batch100', 'worker',
        '--mode', mode, '--start', str(start), '--end', str(end), '--folder', str(folder)]
    run_process(command, folder)
    check_chunk(folder, start, end, operand_sha)
    return {'range': [start, end], 'reused_verified_chunk': False}


def combine(chunks, operand):
    """Reject missing/overlapping decisions before creating any complete-history artifact."""
    result = pd.concat([frame for frame, _ in chunks], ignore_index=True)
    require(result.decision_date.tolist() == operand.date.iloc[400:].tolist() and
            result.first_input_date.tolist() == operand.date.iloc[:-400].tolist() and
            result.last_input_date.tolist() == operand.date.iloc[399:-1].tolist(), 'Assembled coverage differs')
    summary = dict(chunks[0][1]); summary.update(candidate_rows=len(operand), windows=len(result),
        first_decision=result.decision_date.iloc[0], last_decision=result.decision_date.iloc[-1],
        independent_scalar_cells=sum(s['independent_scalar_cells'] for _, s in chunks),
        max_scalar_relative_error=max(s['max_scalar_relative_error'] for _, s in chunks),
        year_counts=result.groupby(result.decision_date.str[:4]).size().to_dict(),
        original_full_pipeline_failures=[chunks[0][1]['original_full_pipeline_failures'][0],
            chunks[-1][1]['original_full_pipeline_failures'][-1]])
    summary['producer_column_comparisons'] = len(result)*1059
    summary['producer_independent_derived_comparisons'] = len(result)*1034
    summary['base25_comparisons_reuse_original_function_and_library'] = True
    return result, summary


def assemble(mode, doc):
    target = DIRECTORY/mode/'assembled'; require(not target.exists(), 'Immutable assembled output exists')
    paths = [existing(mode, a, b, doc['operand_sha256']) for a, b in doc['ranges']]
    require(all(p is not None for p in paths), 'Incomplete recovery chunks')
    chunks = [check_chunk(p, a, b, doc['operand_sha256']) for p, (a, b) in zip(paths, doc['ranges'], strict=True)]
    frame, summary = combine(chunks, pd.read_parquet(study.DIRECTORY/'ETF-input.parquet'))
    with study.network_blocked(): diag = study.diagnostics()
    target.mkdir(); frame.to_parquet(target/'feature-prefix.parquet', index=False)
    save(target/'summary.json', summary); save(target/'diagnostics.json', diag)
    require(read(target/'diagnostics.json') == diag, 'Recovery diagnostic JSON differs')
    receipt = {'status': 'ok', 'mode': mode, 'chunks': len(paths), 'windows': len(frame),
        'chunk_receipt_sha256': {str(p/'completed.json'): file_sha(p/'completed.json') for p in paths},
        'output_sha256': {n: file_sha(target/n) for n in study.OUTPUTS}}
    save(target/'assembly.json', receipt)
    return receipt


def run(workers=4):
    require(1 <= workers <= 8, 'Worker count outside bound'); doc = validate()
    # A separate local recovery lock avoids holding the global data writer during calculation.
    with FileLock(str(DIRECTORY/'orchestrator.lock'), timeout=0):
        for mode in ('study', 'replay'):
            assembled = DIRECTORY/mode/'assembled/assembly.json'
            if assembled.exists():
                for n, sha in read(assembled)['output_sha256'].items():
                    require(file_sha(assembled.parent/n) == sha, 'Existing assembled output changed')
                continue
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_chunk, mode, a, b, doc['operand_sha256']) for a, b in doc['ranges']]
                for future in as_completed(futures):
                    print(json.dumps({'mode': mode, **future.result()}), flush=True)
            print(json.dumps(assemble(mode, doc)), flush=True)
        return offline()


def offline():
    doc = validate(); hashes = {}
    for mode in ('study', 'replay'):
        target = DIRECTORY/mode/'assembled'; assembly = read(target/'assembly.json')
        require(assembly['windows'] == 4852 and assembly['chunks'] == 49, 'Longest assembly incomplete')
        for name, sha in assembly['chunk_receipt_sha256'].items():
            require(file_sha(name) == sha, 'Assembly chunk receipt changed')
        for a, b in doc['ranges']: require(existing(mode, a, b, doc['operand_sha256']) is not None, 'Fresh chunk missing')
        actual = {n: file_sha(target/n) for n in study.OUTPUTS}
        require(actual == assembly['output_sha256'], 'Assembled bytes changed'); hashes[mode] = actual
    require(hashes['study'] == hashes['replay'], 'Fresh replay bytes differ')
    pd.testing.assert_frame_equal(pd.read_parquet(DIRECTORY/'study/assembled/feature-prefix.parquet'),
        pd.read_parquet(DIRECTORY/'replay/assembled/feature-prefix.parquet'), check_exact=True)
    proof = {'status': 'match', 'differences': 0, 'fresh_process_per_chunk': True,
        'network_blocked': True, 'windows_per_mode': 4852, 'chunks_per_mode': 49,
        'output_sha256': hashes['study'], 'recovery_prepared_sha256': file_sha(DIRECTORY/'prepared.json'),
        'new_trading_backtests': 0, 'original_strategy_complete': False}
    save(DIRECTORY/'offline-verification.json', proof); return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker', 'offline'))
    parser.add_argument('--mode', choices=('study', 'replay')); parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int); parser.add_argument('--folder', type=Path)
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    result = worker(args.mode, args.start, args.end, args.folder) if args.action == 'worker' else \
        run(args.workers) if args.action == 'run' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False), flush=True)
