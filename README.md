# Time Variability Correction (TVC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10212122.svg)](https://doi.org/10.5281/zenodo.10212122)

**A Python implementation of Time Variability Correction for climate-model time series.**

This repository contains the TVC method, verification metrics, a case-study workflow, and research scripts used in Shao et al. (2024). TVC aims to improve the representation of temporal variability in climate-model projections across multiple time scales.

## Scientific background

Climate models may reproduce long-term changes while misrepresenting variability at daily-to-seasonal time scales. TVC decomposes a time series into components at multiple temporal scales and adjusts the mean and covariance of those components using an observational reference.

For methodology, assumptions, evaluation and scientific context, consult:

> Shao, Y., Bishop, C. H., Hobeichi, S., Nishant, N., Abramowitz, G., & Sherwood, S. (2024). *Time variability correction of CMIP6 climate change projections*. Journal of Advances in Modeling Earth Systems. https://doi.org/10.1029/2023MS003640

## Repository guide

| File | Purpose |
|---|---|
| `TVC_class.py` | Core `TVC` class, including multiscale decomposition and covariance-based correction. |
| `statistics_tool.py` | Verification metrics, including variance, lag correlation and warm-spell duration index calculations. |
| `TVC_case_study.py` | Case-study workflow demonstrating use of the core class and metrics. |
| `Figure*.py`, `FigureS*.py` | Scripts for generating figures and supplementary analyses in the paper. |
| `AR1_standard_Gaussian_*.py` | Synthetic AR(1) experiments used in the research analysis. |
| `Data/` | Selected input data and intermediate results used by the research scripts. |
| `examples/` | Lightweight synthetic-data example, without the research data dependency. |
| `tests/` | Focused automated tests for selected numerical operations. |

The original research scripts and data remain in their existing locations to preserve the published workflow. The synthetic example is a quick software demonstration, not a replacement for the paper's experiments.

## Installation

Python 3.9 or later is recommended.

```bash
git clone https://github.com/yawenshao/Time-Variability-Correction.git
cd Time-Variability-Correction
python -m pip install -e .
```

For tests and the synthetic demonstration:

```bash
python -m pip install -e ".[test,demo]"
```

Some paper-reproduction scripts additionally use xarray, matplotlib, seaborn and other scientific Python packages. Install the optional research dependencies with:

```bash
python -m pip install -e ".[research]"
```

## Quick start

Run the self-contained synthetic demonstration:

```bash
python examples/synthetic_demo.py
```

The script generates synthetic daily time series, fits TVC using a synthetic observational reference and model training series, and applies the fitted correction to an independent model series. It saves a diagnostic plot in `outputs/`. No external climate data are needed.

This example demonstrates the software workflow; it is not a scientific validation or reproduction of the published CMIP6 experiment.

## Core API

The current implementation accepts one-dimensional time series:

```python
from TVC_class import TVC

scales = [365, 183, 92, 46, 23, 12, 6, 3, 2]

model = TVC(
    obs=observational_training_series,
    raw_train=model_training_series,
    raw_val=model_validation_series,
    scales=scales,
)

corrected_components = model.TVC_postprocess()
corrected_series = corrected_components.sum(axis=1)
```

The observation and model training series must have matching lengths. Choose temporal scales appropriate to the sampling frequency and scientific question. The rolling decomposition omits an initial spin-up segment determined by the selected scales; consult `TVC_class.py` when aligning corrected output with the original time coordinate.

## Reproducing the paper

The root-level `Figure*.py` and `FigureS*.py` scripts document the analyses and plotting workflows used in the article. They may require larger datasets, intermediate files and additional packages listed in their imports. The files in `Data/` are selected research assets and may not include every input used in the study.

- Zenodo software/data record: https://doi.org/10.5281/zenodo.10212122
- Research article: https://doi.org/10.1029/2023MS003640

## Tests

Run the focused unit tests with:

```bash
python -m pytest
```

These tests cover selected numerical operations and array behaviour. They do not constitute complete validation of every scientific use case or reproduce all paper results.

## Limitations

- TVC is a research method; suitability depends on the data and intended application.
- Users should check temporal alignment, missing values, sampling frequency and scale selection.
- The core class operates on one-dimensional time series. It does not provide a complete CF-compliant NetCDF ingestion, metadata-preservation or climate-data publication pipeline.
- The synthetic example is a software demonstration, not evidence that TVC improves every variable, region or climate model.

## Citation

Please cite the associated article and Zenodo record when using this software:

- Shao et al. (2024): https://doi.org/10.1029/2023MS003640
- Zenodo software/data record: https://doi.org/10.5281/zenodo.10212122

The archived Zenodo release should be treated as the citable snapshot associated with the paper.
