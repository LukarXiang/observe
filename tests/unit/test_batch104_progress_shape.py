"""Archive return paths must not mask changed checkpoint identities or saved values."""
import copy

import pytest

from scripts.recover_strategy_batch104_formal_readback import verify_progress_shape


def fixture():
    summary = {'checkpoint':'checkpoint104','catalog_id':'catalog104','manually_reviewed':294}
    latest = {'checkpoint':'checkpoint104','directory':'data/catalog/strategies/progress/checkpoint104'}
    return {**summary,'output':latest['directory']},latest,summary


def test_archive_output_path_is_checked_separately_from_exact_saved_values():
    progress,latest,summary = fixture(); before = copy.deepcopy((progress,latest,summary))
    verify_progress_shape(progress,latest,summary)
    assert (progress,latest,summary) == before


@pytest.mark.parametrize('mode',['extra','missing_output','directory','checkpoint','same_count_other_catalog','count'])
def test_only_known_archive_field_can_differ_and_identities_cannot_be_ignored(mode):
    progress,latest,summary = fixture()
    if mode == 'extra': progress['unknown'] = True
    elif mode == 'missing_output': progress.pop('output')
    elif mode == 'directory': progress['output'] = 'another/directory'
    elif mode == 'checkpoint': latest['checkpoint'] = 'other_checkpoint'
    elif mode == 'same_count_other_catalog': summary['catalog_id'] = 'other_catalog'
    else: summary['manually_reviewed'] = 293
    with pytest.raises(ValueError): verify_progress_shape(progress,latest,summary)
