"""Time Variability Correction (TVC) for climate-model time series."""

from tvc.constants import DEFAULT_SCALES, SCALE_LABELS, spin_up
from tvc.core import TVC
from tvc.metrics import Statistics, statistics

__version__ = "0.2.0"

__all__ = ["TVC", "Statistics", "statistics", "DEFAULT_SCALES", "SCALE_LABELS", "spin_up", "__version__"]
