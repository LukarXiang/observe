"""Actual window equivalence, interrupted output retention and chunk corruption checks."""
import sys

import pandas as pd
import pytest

from scripts import recover_strategy_batch100 as recovery
from scripts import research_strategy_batch100 as study
from scripts.run_strategy_batch4 import read


def test_all_4852_windows_are_partitioned_without_gap():
    plan = recovery.ranges(4852)
    assert len(plan) == 49 and plan[0] == (0, 100) and plan[-1] == (4800, 4852)
    assert [i for a, b in plan for i in range(a, b)] == list(range(4852))


def test_timeout_retains_streamed_progress_and_terminal_receipt(tmp_path):
    command = [sys.executable, '-u', '-c', 'import time;print("first-window",flush=True);time.sleep(10)']
    with pytest.raises(ValueError, match='terminal receipt retained'):
        recovery.run_process(command, tmp_path, timeout=.5)
    terminal = read(tmp_path/'command.json')
    assert terminal['timed_out'] and terminal['returncode'] != 0
    assert (tmp_path/'stdout.log').read_text() == 'first-window\n'
    assert terminal['stdout_sha256'] == study.file_sha(tmp_path/'stdout.log')


@pytest.fixture(scope='module')
def actual_chunks():
    frame = pd.read_parquet(study.DIRECTORY/'ETF-input.parquet').iloc[:404].copy()
    whole = study.components(frame)
    chunks = [recovery.calculate(frame, 0, 2), recovery.calculate(frame, 2, 4)]
    return frame, whole, chunks


def test_partitioning_preserves_every_original_cell_and_window_seed(actual_chunks):
    operand, (expected, summary), chunks = actual_chunks
    actual, merged = recovery.combine(chunks, operand)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for name, value in summary.items(): assert merged[name] == value
    assert merged['producer_independent_derived_comparisons'] == 4*1034
    assert merged['base25_comparisons_reuse_original_function_and_library']


@pytest.mark.parametrize('mode', ['missing', 'overlap', 'future_cutoff'])
def test_assembly_rejects_missing_duplicate_or_future_input(actual_chunks, mode):
    operand, _, chunks = actual_chunks
    broken = [(f.copy(), dict(s)) for f, s in chunks]
    if mode == 'missing': broken = broken[:-1]
    elif mode == 'overlap': broken.append(broken[-1])
    else: broken[0][0].loc[0, 'last_input_date'] = broken[0][0].loc[0, 'decision_date']
    with pytest.raises(ValueError, match='coverage differs'): recovery.combine(broken, operand)


def test_chunk_bytes_and_operand_are_bound_before_reuse(actual_chunks, tmp_path):
    _, _, chunks = actual_chunks; frame, summary = chunks[0]
    frame.to_parquet(tmp_path/'feature-prefix.parquet', index=False); study.save(tmp_path/'summary.json', summary)
    study.save(tmp_path/'completed.json', {'status': 'ok', 'range': [0, 2], 'operand_sha256': 'input',
        'network_blocked': True, 'sha256': {n: study.file_sha(tmp_path/n) for n in ('feature-prefix.parquet', 'summary.json')}})
    recovery.check_chunk(tmp_path, 0, 2, 'input')
    with pytest.raises(ValueError, match='identity'): recovery.check_chunk(tmp_path, 0, 2, 'other-input')
    frame.loc[0, 'volume'] *= 100; frame.to_parquet(tmp_path/'feature-prefix.parquet', index=False)
    with pytest.raises(ValueError, match='artifact changed'): recovery.check_chunk(tmp_path, 0, 2, 'input')


def test_chunk_cannot_exceed_available_400_row_window(actual_chunks):
    operand, _, _ = actual_chunks
    with pytest.raises(ValueError, match='outside input range'): recovery.calculate(operand, 0, 5)
