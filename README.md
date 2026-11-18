# Time Variability Correction (TVC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10212122.svg)](https://doi.org/10.5281/zenodo.10212122)

**A Python implementation of Time Variability Correction for climate-model time series.**

This repository contains the TVC method, verification metrics, an interactive demo dashboard, and the research scripts used in Shao et al. (2024). TVC aims to improve the representation of temporal variability in climate-model projections across multiple time scales.

## Scientific background

Climate models may reproduce long-term changes while misrepresenting variability at daily-to-seasonal time scales. TVC decomposes a time series into components at multiple temporal scales and adjusts the mean and covariance of those components using an observational reference.

For methodology, assumptions, evaluation and scientific context, consult:

> Shao, Y., Bishop, C. H., Hobeichi, S., Nishant, N., Abramowitz, G., & Sherwood, S. (2024). *Time variability correction of CMIP6 climate change projections*. Journal of Advances in Modeling Earth Systems. https://doi.org/10.1029/2023MS003640

## Repository layout

| Path | Purpose |
|---|---|
| `src/tvc/core.py` | Core `TVC` class: multiscale decomposition and covariance-based correction. |
| `src/tvc/metrics.py` | `statistics` (alias `Statistics`): variance, lag correlation and warm-spell duration index. |
| `src/tvc/constants.py`, `io.py`, `plotting.py`, `synthetic.py` | Shared constants, `Data/` path helpers, figure styling, AR(1) generator. |
| `app/` | Streamlit + Plotly demo dashboard. |
| `scripts/paper/` | Figure and analysis scripts from the paper (see its [README](scripts/paper/README.md)). |
| `examples/` | Lightweight synthetic-data example with no research-data dependency. |
| `Data/` | Selected input data and intermediate results used by the research scripts and the dashboard. |
| `tests/` | Unit tests and a regression test on the shipped site case study. |

The TVC and metric implementations are unchanged from the published version; this release reorganises the repository around them.

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/yawenshao/Time-Variability-Correction.git
cd Time-Variability-Correction
python -m pip install -e .
```

Optional extras:

| Extra | Installs | Use |
|---|---|---|
| `demo` | matplotlib | `examples/synthetic_demo.py` |
| `dashboard` | streamlit, plotly, xarray, netCDF4 | `app/streamlit_app.py` |
| `research` | xarray, netCDF4, matplotlib, seaborn, cartopy | `scripts/paper/` |
| `test` | pytest, pytest-cov | test suite |
| `dev` | all of the above plus ruff, mypy, pre-commit | development |

```bash
python -m pip install -e ".[dashboard]"
```

## Demo dashboard

```bash
python -m pip install -e ".[dashboard]"
streamlit run app/streamlit_app.py
```

It has four tabs:

- **Site case study**: ERA5 versus any of 23 CMIP6 models at -37.5°, 144.5° (near Melbourne). TVC runs live, and the tab shows metric cards (variance, lag-1, 5-day lag, WSDI), the scale-by-scale covariance matrices for observations, raw and corrected series, variance by time scale, and time series per component. Switch between in-sample (as in the paper) and out-of-sample evaluation (fit on 1950–1981, apply to 1982–2014).
- **Multi-model comparison**: an interactive Figure 3 plus a table of each model's percentage improvement towards ERA5.
- **Global view**: maps and per-model box plots of percentage improvement for each scenario (Figures 4–5), and 1° TVC − raw difference maps (Figure 8 / S8).
- **Synthetic playground**: sliders for AR(1) reference and model processes, showing out-of-sample variance by scale and the autocorrelation function.

## Quick start

```python
from tvc import TVC, DEFAULT_SCALES, spin_up

model = TVC(
    obs=observational_training_series,
    raw_train=model_training_series,
    raw_val=model_validation_series,
    scales=DEFAULT_SCALES,  # [365, 183, 92, 46, 23, 12, 6, 3, 2]
)

corrected_components = model.TVC_postprocess()
corrected_series = corrected_components.sum(axis=1)
# The first spin_up(DEFAULT_SCALES) samples of the input are not returned.
```

The observation and model training series must have matching lengths. Choose temporal scales appropriate to the sampling frequency and scientific question. The legacy imports `from TVC_class import TVC` and `from statistics_tool import statistics` still work but emit a `DeprecationWarning`.

Self-contained synthetic example (saves a plot to `outputs/`):

```bash
python -m pip install -e ".[demo]"
python examples/synthetic_demo.py
```

## Reproducing the paper

See [`scripts/paper/README.md`](scripts/paper/README.md) for the script-to-figure mapping and for inputs that are not included in `Data/`.

- Zenodo software/data record: https://doi.org/10.5281/zenodo.10212122
- Research article: https://doi.org/10.1029/2023MS003640

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install
python -m pytest            # unit + site regression tests
python -m pytest app        # dashboard smoke test
ruff check .
```

The tests cover selected numerical operations and a site regression; they do not constitute complete validation of every scientific use case.

## Limitations

- TVC is a research method; suitability depends on the data and intended application.
- Users should check temporal alignment, missing values, sampling frequency and scale selection.
- The core class operates on one-dimensional time series. It does not provide a complete CF-compliant NetCDF ingestion, metadata-preservation or climate-data publication pipeline.
- The synthetic example and dashboard are demonstrations, not evidence that TVC improves every variable, region or climate model.

## Citation

Please cite the associated article and Zenodo record when using this software:

- Shao et al. (2024): https://doi.org/10.1029/2023MS003640
- Zenodo software/data record: https://doi.org/10.5281/zenodo.10212122

The archived Zenodo release should be treated as the citable snapshot associated with the paper.
