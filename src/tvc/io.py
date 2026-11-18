"""Path helpers and loaders for the research data shipped in ``Data/``.

The data and figure directories default to ``<repo>/Data`` and ``<repo>/Figures``
and can be overridden with the ``TVC_DATA_DIR`` and ``TVC_FIGURES_DIR``
environment variables.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir():
    return Path(os.environ.get("TVC_DATA_DIR", REPO_ROOT / "Data"))


def figures_dir():
    path = Path(os.environ.get("TVC_FIGURES_DIR", REPO_ROOT / "Figures"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(name):
    return data_dir() / name


def figure_path(name):
    return figures_dir() / name


def load_csv_array(name, flatten=True):
    """Read a header-less CSV from ``Data/`` into a NumPy array."""
    values = np.array(pd.read_csv(data_path(name), header=None))
    return values.flatten() if flatten else values


def load_model_names():
    """Return (model ids, institution-qualified model names) used across the paper."""
    models = np.array(pd.read_csv(data_path("model_name_ssp126585.csv"), header=None).iloc[:, 0])
    names = np.array(pd.read_csv(data_path("model_institution_ssp126585.csv"), header=None).iloc[:, 1])
    return models, names


def open_dataset(name):
    import xarray as xr

    return xr.open_dataset(data_path(name))
