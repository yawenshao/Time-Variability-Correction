"""Tests for the dashboard's data preparation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboard_lib as lib  # noqa: E402


def test_box_stats_whiskers_ignore_extreme_outliers():
    values = np.concatenate([np.linspace(50, 100, 99), [-876.0]])
    stats = lib.box_stats(values)
    assert stats["lowerfence"] >= 40  # the -876 outlier does not stretch the whisker
    assert stats["upperfence"] == 100
    assert stats["q1"] < stats["median"] < stats["q3"]


def test_box_stats_matches_matplotlib():
    import matplotlib.cbook as cbook

    values = np.random.default_rng(1).normal(60, 30, 648)
    ref = cbook.boxplot_stats(values)[0]
    stats = lib.box_stats(values)
    assert np.isclose(stats["median"], ref["med"])
    assert np.isclose(stats["lowerfence"], ref["whislo"])
    assert np.isclose(stats["upperfence"], ref["whishi"])


def test_prct_box_table_medians_are_positive_for_paper_data():
    table = lib.prct_box_table("Variance", ["historical", "ssp585"])
    assert len(table) == 2 * 23
    assert (table["median"] > 0).all()


def test_coastlines_load():
    lon, lat = lib.load_coastlines()
    assert len(lon) == len(lat) > 1000
