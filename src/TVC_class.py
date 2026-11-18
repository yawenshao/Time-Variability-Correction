"""Deprecated import location kept for the published workflow; use ``from tvc import TVC``."""

import warnings

from tvc.core import TVC

warnings.warn(
    "'TVC_class' is deprecated; use 'from tvc import TVC' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TVC"]
