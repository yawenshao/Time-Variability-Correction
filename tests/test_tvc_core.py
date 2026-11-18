"""Tests for the TVC core numerical operations."""

import numpy as np

from tvc import TVC, spin_up
from tvc.synthetic import generate_ar1


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
    assert components.shape == (n - spin_up(scales), len(scales) + 1)
    assert np.isfinite(components).all()
    np.testing.assert_allclose(components.sum(axis=1), series[spin_up(scales) :])


def test_in_sample_correction_reproduces_observed_mean_and_covariance():
    scales = [60, 30, 15, 7, 3, 2]
    obs = generate_ar1(3000, 0.8, 1) + 20.0
    raw = 1.5 * generate_ar1(3000, 0.4, 2) + 18.0
    model = TVC(obs, raw, raw, scales)
    corrected = np.real(model.TVC_postprocess())

    obs_mean, obs_cov = model.stat_obs(model.get_decomposition("obs"))
    corr_mean, corr_cov = model.stat_obs(corrected)
    np.testing.assert_allclose(corr_mean, obs_mean, atol=1e-8)
    np.testing.assert_allclose(corr_cov, obs_cov, atol=1e-6)


def test_get_tvc_covariance_matches_stat_obs():
    scales = [20, 10, 5, 2]
    series = generate_ar1(800, 0.6, 3)
    model = TVC(series, series, series, scales)
    _, expected = model.stat_obs(model.get_decomposition("train"))
    np.testing.assert_allclose(model.get_TVC_covariance("train"), expected)
