"""Focused tests for selected TVC numerical operations."""
import numpy as np
from TVC_class import TVC


def test_stat_obs_returns_expected_mean_and_sample_covariance():
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    model = TVC(values[:, 0], values[:, 0], values[:, 0], scales=[2, 1])
    mean, covariance = model.stat_obs(values)
    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(covariance, [[1.0, 2.0], [2.0, 4.0]])


def test_multiscale_decomposition_shape_and_reconstruction():
    rng = np.random.default_rng(42)
    n = 1200
    series = rng.normal(size=n)
    scales = [30, 15, 7, 3, 2]
    model = TVC(series, series, series, scales)
    components = model.sep_time_scales_rolling("val")
    spin_up = sum(scales) - len(scales)
    assert components.shape == (n - spin_up, len(scales) + 1)
    assert np.isfinite(components).all()
    np.testing.assert_allclose(components.sum(axis=1), series[spin_up:])
