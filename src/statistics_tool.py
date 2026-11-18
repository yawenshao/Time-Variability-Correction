"""Deprecated import location kept for the published workflow; use ``from tvc import Statistics``."""

import warnings

from tvc.metrics import Statistics, statistics

warnings.warn(
    "'statistics_tool' is deprecated; use 'from tvc import Statistics' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["statistics", "Statistics"]
