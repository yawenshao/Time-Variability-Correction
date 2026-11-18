"""Regression test on the shipped ERA5 / ACCESS-ESM1-5 site case study (-37.5, 144.5)."""

import numpy as np
import pytest

from tvc import DEFAULT_SCALES, TVC, spin_up, statistics
from tvc.io import data_path, load_csv_array

OBS_FILE = "ERA5_obs_-37.5_144.5_tasmax_hist.csv"
RAW_FILE = "ERA5_ACCESS-ESM1-5_-37.5_144.5_tasmax_hist.csv"

pytestmark = pytest.mark.skipif(not data_path(OBS_FILE).exists(), reason="site case-study data not available")


def test_site_case_study_metrics_are_stable():
    obs = load_csv_array(OBS_FILE)
    raw = load_csv_array(RAW_FILE)
    corrected = np.real(TVC(obs, raw, raw, DEFAULT_SCALES).TVC_postprocess().sum(axis=1))
    init = spin_up(DEFAULT_SCALES)

    tvc_stats = statistics(corrected)
    assert tvc_stats.sample_var_30y_mv() == pytest.approx(41.0699, abs=1e-3)
    assert tvc_stats.lag_corr_nonan_notr_30y_mv(1, 1) == pytest.approx(0.82959, abs=1e-4)
    assert tvc_stats.lag_corr_nonan_notr_30y_mv(5, 1) == pytest.approx(0.80543, abs=1e-4)

    raw_stats = statistics(raw[init:])
    obs_stats = statistics(obs[init:])
    assert raw_stats.sample_var_30y_mv() == pytest.approx(49.3058, abs=1e-3)
    assert obs_stats.sample_var_30y_mv() == pytest.approx(41.0500, abs=1e-3)
