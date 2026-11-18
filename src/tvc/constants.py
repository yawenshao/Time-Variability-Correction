"""Shared constants used by the paper scripts and the dashboard."""

DEFAULT_SCALES = [365, 183, 92, 46, 23, 12, 6, 3, 2]

SCALE_LABELS = [
    "365-day",
    "183-day",
    "92-day",
    "46-day",
    "23-day",
    "12-day",
    "6-day",
    "3-day",
    "2-day",
    "1-day",
]

DAYS_30Y = 10957

SITE_LAT, SITE_LON = -37.5, 144.5
SITE_START, SITE_END = "1950-01-01", "2014-12-31"
WSDI_BASE_START, WSDI_BASE_END = 1961, 1990
WSDI_YEAR_START, WSDI_YEAR_END = 1952, 2014

SCENARIOS = ["historical", "ssp126", "ssp585"]


def spin_up(scales):
    """Number of leading samples dropped by the rolling multiscale decomposition."""
    return sum(scales) - len(scales)
