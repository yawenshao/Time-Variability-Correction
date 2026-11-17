"""Tests for selected verification-statistics functionality."""
import numpy as np
from statistics_tool import statistics


def test_nday_discrete_average_returns_non_overlapping_aggregates():
    values = np.arange(1.0, 9.0)
    result = statistics(values).nday_discrete_average_df(2)
    np.testing.assert_allclose(result, [1.5, 3.5, 5.5, 7.5])
