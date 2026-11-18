"""Tests for the verification-statistics functionality."""

import numpy as np
import pandas as pd

from tvc import statistics
from tvc.synthetic import generate_ar1


def test_nday_discrete_average_returns_non_overlapping_aggregates():
    values = np.arange(1.0, 9.0)
    result = statistics(values).nday_discrete_average_df(2)
    np.testing.assert_allclose(result, [1.5, 3.5, 5.5, 7.5])


def test_lag1_correlation_and_variance_match_ar1_theory():
    phi = 0.7
    series = generate_ar1(40 * 365, phi, 5)
    stats = statistics(series)
    assert abs(stats.lag_corr_nonan_notr_30y_mv(1, 1) - phi) < 0.03
    assert abs(stats.sample_var_30y_mv() - 1.0 / (1.0 - phi**2)) < 0.15


def test_base_period_percentile_returns_one_value_per_calendar_day():
    years = (1961, 1990)
    n_days = len(pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    base = np.random.default_rng(0).normal(size=n_days)
    prcts = statistics(base).base_period_percentile(base, years[0], years[1], 5)
    assert prcts.shape == (365,)
    assert np.isfinite(prcts).all()


def test_wsdi_counts_only_spells_of_at_least_n_days():
    dates = pd.date_range("2001-01-01", "2002-12-31")
    data = np.zeros(len(dates))
    # 2001: one 7-day spell (counted); 2002: one 5-day spell (too short for nday=6).
    data[(dates >= "2001-03-01") & (dates <= "2001-03-07")] = 10.0
    data[(dates >= "2002-06-01") & (dates <= "2002-06-05")] = 10.0
    prcts = np.ones(365)

    wsdi = statistics(data).WSDI(6, False, prcts, 2001, 2002)
    np.testing.assert_array_equal(wsdi, [7.0, 0.0])


def test_wsdi_masks_years_with_too_many_missing_days():
    dates = pd.date_range("2001-01-01", "2002-12-31")
    data = np.zeros(len(dates))
    data[(dates >= "2002-02-01") & (dates <= "2002-02-05")] = np.nan

    wsdi = statistics(data).WSDI(6, False, np.ones(365), 2001, 2002)
    assert wsdi[0] == 0.0
    assert np.isnan(wsdi[1])
