"""The legacy module names must keep working for the published workflow."""

import importlib
import sys

import pytest

import tvc


@pytest.mark.parametrize(
    ("module", "name", "target"),
    [("TVC_class", "TVC", tvc.TVC), ("statistics_tool", "statistics", tvc.statistics)],
)
def test_legacy_modules_reexport_with_deprecation_warning(module, name, target):
    sys.modules.pop(module, None)
    with pytest.warns(DeprecationWarning):
        legacy = importlib.import_module(module)
    assert getattr(legacy, name) is target
