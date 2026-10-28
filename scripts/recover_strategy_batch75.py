"""Adapt frozen ETF close strings to the numeric original-history interface."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

from scripts import review_strategy_batch75 as batch

SELECTED = batch.selected
COMMANDS = batch.commands
FILES = tuple(sorted(set(batch.FILES) | {
    'scripts/recover_strategy_batch75.py', 'tests/unit/test_batch75_recovery.py'}))


def numeric_selected(directory, number, names, namespace):
    if number == 1 and names == ['get_rank']:
        namespace = namespace.copy()
        history = namespace['attribute_history']

        def numeric_history(*args, **kwargs):
            frame = history(*args, **kwargs).copy()
            # Use the exact conversion already used by the frozen window checks.
            frame['close'] = frame.close.to_numpy(dtype=float)
            return frame

        namespace['attribute_history'] = numeric_history
    return SELECTED(directory, number, names, namespace)


def commands():
    result = COMMANDS()
    result[0].append('scripts/recover_strategy_batch75.py')
    result[1].append('tests/unit/test_batch75_recovery.py')
    return result


def execute(action, root, directory):
    batch.require(action in ('candidate_study', 'offline', 'checks', 'finish'), 'Unsupported recovery action')
    provenance = directory / 'numeric-input-recovery.json'
    if action == 'candidate_study':
        batch.require(not provenance.exists(), 'Recovery already archived')
        failure = directory / 'ranking-input-failure.json'
        batch.require(batch.file_sha('scripts/review_strategy_batch75.py') == batch.read(failure)['driver_sha256'],
            'Failed original driver changed')
        batch.save(provenance, {'status': 'numeric_input_adapter', 'failure': str(failure),
            'failure_sha256': batch.file_sha(failure), 'implementation_sha256': {
                name: batch.file_sha(name) for name in FILES},
            'change': 'Only single get_rank attribute_history close column converted with to_numpy(dtype=float)',
            'original_study_sha256': batch.file_sha(directory / 'component-research.json'),
            'raw_inputs_changed': False, 'formulas_changed': False, 'new_requests': 0, 'not_a_backtest': True})
    proof = batch.read(provenance)
    batch.require(proof['implementation_sha256'] == {name: batch.file_sha(name) for name in FILES} and
        batch.file_sha(proof['failure']) == proof['failure_sha256'] and
        batch.file_sha(directory / 'component-research.json') == proof['original_study_sha256'],
        'Recovery implementation or provenance changed')
    with ExitStack() as stack:
        stack.enter_context(patch.object(batch, 'selected', numeric_selected))
        stack.enter_context(patch.object(batch, 'FILES', FILES))
        stack.enter_context(patch.object(batch, 'commands', commands))
        stack.enter_context(patch.object(batch, 'CORE', batch.CORE | {
            'ranking-input-failure.json', 'numeric-input-recovery.json'}))
        return getattr(batch, action)(root, directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['candidate_study', 'offline', 'checks', 'finish'])
    parser.add_argument('--root', default='data')
    parser.add_argument('--directory', type=Path, default=batch.DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(execute(args.action, args.root, args.directory), ensure_ascii=False, default=str))
