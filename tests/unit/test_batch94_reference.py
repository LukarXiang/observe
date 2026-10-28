"""Independent mathematical and input-provenance boundaries for Batch94."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts import reference_strategy_batch94 as reference
from scripts.review_strategy_batch18 import save
from observe.runs import file_sha


def test_two_point_gaussian_posterior_hand_solution():
    # Precision [[70/3,-10],[-10,50/3]], RHS [40/3,40/3].
    # Inverse determinant is 2600/9. This is independent of any filter recursion.
    means = reference.posterior([1., 2.])
    np.testing.assert_allclose(means, [16/13, 20/13], rtol=0, atol=1e-15)
    np.testing.assert_allclose(reference.posterior_variances(2), [3/52, 21/260], rtol=0, atol=1e-15)


@pytest.mark.parametrize('size', [2, 3, 20])
def test_constant_observations_preserved(size):
    np.testing.assert_allclose(reference.posterior(np.ones(size)*1.7), np.ones(size)*1.7, rtol=0, atol=2e-15)


@pytest.mark.parametrize('observations', [[], [1.], [1., np.nan], [np.inf, 2.]])
def test_invalid_observations_rejected(observations):
    with pytest.raises(ValueError, match='Posterior observations invalid'): reference.posterior(observations)


def test_precision_solves_full_joint_normal_equations():
    y = np.asarray([1., 1.2, .8, 1.5, .9])
    d, off, rhs = reference.precision_system(y)
    matrix = np.diag(d)+np.diag(off, 1)+np.diag(off, -1)
    expected = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(reference.posterior(y), expected, rtol=0, atol=5e-16)
    np.testing.assert_allclose(reference.posterior_variances(len(y)), np.diag(np.linalg.inv(matrix)), rtol=0, atol=2e-17)


def test_smoothing_uses_later_observations_inside_window():
    before = reference.posterior(np.ones(20))
    after = reference.posterior([1.]*19+[2.])
    assert after[0] > before[0]
    assert after[-1] > before[-1]


def test_scalar_slope_known_linear_trend():
    assert reference.independent_slope([1+3*i for i in range(20)]) == 3.


def test_fork_loader_requires_isolated_worker(tmp_path):
    with patch.dict(reference.os.environ, {}, clear=True), pytest.raises(ValueError, match='fresh bytecode-disabled worker'):
        reference.load_fork(tmp_path)


@pytest.mark.parametrize('mutation', ['empty_partitions', 'changed_frame', 'changed_partition', 'valid'])
def test_rooted_index_rejects_self_declared_inputs(tmp_path, mutation):
    frame = pd.DataFrame({'date': ['2026-01-01'], 'high': [1.]})
    bound = {'partitions': [{'file': 'actual.parquet', 'sha256': 'actual'}], 'rows': 1}
    declared = {**bound, 'frozen_sha256': None}; frozen = frame.copy()
    if mutation == 'empty_partitions': declared['partitions'] = []
    if mutation == 'changed_partition': declared['partitions'] = [{'file': 'elsewhere', 'sha256': 'actual'}]
    if mutation == 'changed_frame': frozen['high'] = 1.001
    frozen.to_parquet(tmp_path/'index-input.parquet', index=False)
    declared['frozen_sha256'] = file_sha(tmp_path/'index-input.parquet'); save(tmp_path/'index-binding.json', declared)
    with patch.object(reference.previous, 'canonical_root'), patch.object(reference, 'index_operand', return_value=(frame, bound)):
        if mutation == 'valid': assert reference.validate_index('data', tmp_path) == bound
        else:
            with pytest.raises((ValueError, AssertionError)): reference.validate_index('data', tmp_path)


def test_original_helper_calls_smooth_with_literal_parameters():
    calls = []
    class Spy:
        def __init__(self, **kwargs): calls.append(kwargs)
        def smooth(self, values):
            calls.append('smooth'); return np.asarray(values).reshape(-1, 1), None
    ns = reference.previous.literal(reference.previous.DIRECTORY, 1, {'np': np, 'KalmanFilter': Spy})
    result = ns['kalman_filter'](np.ones(20), .15)
    assert result.shape == (20, 1)
    assert calls == [{'initial_state_mean': 1., 'initial_state_covariance': .15, 'transition_matrices': 1,
        'observation_covariance': .15, 'transition_covariance': .1, 'transition_offsets': None}, 'smooth']


def test_runtime_scope_rejects_platform_equivalence(tmp_path):
    save(tmp_path/'runtime-binding.json', {'fork': 'sktime 0.38.5 unofficial pykalman fork',
        'dependency': 'scikit-base 0.12.5', 'installation_performed': False,
        'original_platform_library_equivalence_proven': True, 'files': []})
    with pytest.raises(ValueError, match='Runtime scope changed'): reference.validate_runtime(tmp_path)
