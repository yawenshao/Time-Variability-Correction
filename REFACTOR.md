# Refactor and dashboard worklog

Record of the repository review, structural refactor, Streamlit demo, and the two follow-up checks (Global view, multi-model lag-1 table). Written so the work can be audited without the chat history.

The scientific method was not changed. `src/tvc/core.py` and `src/tvc/metrics.py` are the original `TVC_class.py` and `statistics_tool.py` moved into a package. Their Python syntax trees match the originals exactly, apart from one alias line (`Statistics = statistics`) at the end of `metrics.py`.

Package version after this work: **0.2.0** (`pyproject.toml`).

---

## 1. Starting point

The repository was a flat research code release for Shao et al. (2024), *Time variability correction of CMIP6 climate change projections*, JAMES, https://doi.org/10.1029/2023MS003640, archived at https://doi.org/10.5281/zenodo.10212122.

Everything lived at the repository root:

| Original file | Role |
|---|---|
| `TVC_class.py` | `TVC` class: rolling multiscale decomposition, component mean and covariance, covariance correction via matrix square roots |
| `statistics_tool.py` | `statistics` class: n-day averages, 30-year-mean-relative variance and lag correlation, ETCCDI-style warm spell duration index (WSDI) |
| `TVC_case_study.py` | One-site workflow on ERA5 and ACCESS-ESM1-5 at −37.5°, 144.5° |
| `Figure*.py`, `FigureS*.py` | Paper figure scripts |
| `AR1_standard_Gaussian_nocycle_30_500years_check_statistics.py` | Synthetic AR(1) experiment, 100 realisations |
| `examples/synthetic_demo.py` | Small synthetic demo, no climate data |
| `tests/test_tvc_core.py`, `tests/test_statistics.py` | Two numerical tests |
| `pyproject.toml` | Installable as root modules `TVC_class` and `statistics_tool` |
| `.github/workflows/tests.yml` | pytest on Python 3.10, 3.11, 3.12 |
| `Data/` | Site CSVs, model lists, precomputed NetCDF metric fields, AR(1) result CSVs |

`pyproject.toml` already declared core dependencies (`numpy`, `pandas`, `scipy`) and extras `demo`, `research`, `test`. There was no package, no linter, no type-check config, and no dashboard.

The local machine used for the work: Windows, PowerShell, Python 3.14.6, NumPy 2.4.6, pandas 3.0.3, SciPy 1.18.0. Streamlit 1.58.0, Plotly 6.7.0 and xarray 2026.7.0 were already installed. `netCDF4` was not, so xarray could not open the `.nc` files until it was installed. Git was not on `PATH`, so file comparisons used Python's `ast` module rather than `git diff`.

---

## 2. Decisions agreed before coding

Two choices were made up front, then a constraint was added.

**Dashboard: Streamlit + Plotly.** Fastest way to get an interactive demo. Panel and Dash were the alternatives that were not used.

**Refactor scope: proper `src/` package, paper scripts moved, outputs kept.** The published scripts were not rewritten. A full config-driven pipeline was not built.

**Constraint added when implementation started: do not change function logic.** Several things in the first review looked like bugs. They were left alone because the existing behaviour was described as intended. Section 4 lists each one and what was done instead.

---

## 3. Review findings

These were the structural and tooling problems the refactor was meant to fix. They are separate from the "looks like a bug but leave the logic" list in section 4.

- The two scientific modules were installed as loose root modules. The class `statistics` collides in name with the Python standard-library module `statistics`.
- Paper scripts repeated the same font block, the scale list `[365, 183, 92, 46, 23, 12, 6, 3, 2]`, the spin-up length `sum(scales) - len(scales)`, the site dates `1950-01-01` to `2014-12-31`, and hard-coded `./Data` and `./Figures` paths.
- `Figures/` did not exist, so a first run of any figure script failed at `savefig` unless the folder had been created by hand.
- Figure 6 (`Figure6_global_aggregate_scatter_plots.py`) and Figure 7 (`Figure7_one_model_lat_aggregate_heatmap.py`) open `Data/Var_all_global_{historical,ssp585}_tasmax_{raw,TVC}.nc`. Those four files are not in the repository.
- Figure 8 (`Figure8_S8_global_map_plot_6plots.py`) needs `Data/World_Continents.shp`, which is also not in the repository. The script's own comment points at https://hub.arcgis.com/datasets/CESJ::world-continents/explore.
- Tests covered one covariance calculation, one decomposition shape check, and one 2-day rolling average. Nothing covered WSDI, lag correlation, a real site series, or the legacy import names.
- CI did not run on Python 3.9 even though `requires-python` is `>=3.9`, and it did not lint.

---

## 4. Things that looked wrong and were deliberately left

The first review proposed "fixing" these. After the instruction to keep function logic, each one was checked and left as published behaviour.

### 4.1 `base_period_percentile` reshape size

In `statistics.base_period_percentile` the calendar-day window array is reshaped with

```python
data_nday.reshape((base_st - base_end + 1, 365, win_len))
```

`base_st - base_end + 1` is negative (for 1961–1990 it is −28). In NumPy, one negative dimension means "infer this size". Confirmed on the local NumPy 2.4.6: `np.arange(12).reshape((-2, 3, 2)).shape` is `(2, 3, 2)`. The call therefore reshapes to `(n_years, 365, win_len)` and is not a bug. It was not changed. A test (`test_base_period_percentile_returns_one_value_per_calendar_day`) locks the result shape at `(365,)`.

### 4.2 Length check via `assert` and `exit()`

`TVC.__init__` still does `try: assert len(obs) == len(raw_train) / except: print(...); exit()`. Replacing this with `ValueError` would change control flow for callers, so it stayed.

### 4.3 Unknown series name in `sep_time_scales_rolling`

If `var` is not `'obs'`, `'train'` or `'val'`, `data` is never assigned and the method fails with `UnboundLocalError`. No guard was added.

### 4.4 Complex values from `sqrtm`

`bias_correction` can return a complex array because `scipy.linalg.sqrtm` does. Callers take the real part (`np.real` in the dashboard, `np.real_if_close` in `examples/synthetic_demo.py`). The core method still returns whatever `sqrtm` returns.

### 4.5 `statistics` methods mutate `self.data`

`WSDI` replaces `self.data` with `self.data[-len(self.dates):]` and sets `self.dates`, `self.yr_st`, `self.yr_end`, `self.year_mis`. That mutation is part of how the later methods (`move_window_prct_op`, `consecutive_day_filter`) see the series. It was not refactored into a pure function.

### 4.6 Hard-coded 30-year window

`lag_corr_nonan_notr_30y_mv` uses `10957 // win` and `sample_var_30y_mv` uses `10957`. That is 30 × 365.25, rounded. It was not parameterised. `DAYS_30Y = 10957` in `src/tvc/constants.py` documents the number; the methods do not read that constant, so editing the constant does not change the metric.

### 4.7 In-sample use of the same series for training and validation

`TVC_case_study.py`, Figure 1–3 and the dashboard's default site mode all construct `TVC(obs, raw, raw, scales)`. That is the paper's in-sample setup, not a mistake. The dashboard adds a separate out-of-sample mode on top (section 8.4); it does not replace this.

---

## 5. New layout

```
src/
  tvc/
    __init__.py       public exports and __version__ = "0.2.0"
    core.py           TVC class, logic unchanged
    metrics.py        statistics class, logic unchanged, plus Statistics = statistics
    constants.py      scales, labels, site dates, WSDI years, spin_up()
    io.py             Data/ and Figures/ paths, CSV and model-name loaders
    plotting.py       apply_paper_style(size)
    synthetic.py      generate_ar1()
  TVC_class.py        deprecated re-export of TVC
  statistics_tool.py  deprecated re-export of statistics and Statistics
scripts/paper/        moved figure and experiment scripts
app/
  streamlit_app.py    dashboard UI
  dashboard_lib.py    data preparation, no Streamlit import
  test_app.py         Streamlit AppTest smoke test
  test_dashboard_lib.py
  assets/coastlines_110m.json
tests/                package tests (run by default pytest)
examples/synthetic_demo.py
Data/                 untouched
```

`pyproject.toml` installs this with

```toml
[tool.setuptools]
package-dir = { "" = "src" }
py-modules = ["TVC_class", "statistics_tool"]

[tool.setuptools.packages.find]
where = ["src"]
```

so `import tvc`, `import TVC_class` and `import statistics_tool` all work from an editable install. Pytest is configured with `pythonpath = ["src"]` and `testpaths = ["tests"]`. The dashboard tests live under `app/` and are not part of the default `pytest` run; CI runs them in a separate job (`python -m pytest -q app/test_app.py`). `app/test_dashboard_lib.py` is run the same way when the app tests are invoked as `pytest app`.

### 5.1 What `tvc` exports

`src/tvc/__init__.py` exports `TVC`, `Statistics`, `statistics`, `DEFAULT_SCALES`, `SCALE_LABELS`, `spin_up`, `__version__`.

`DEFAULT_SCALES = [365, 183, 92, 46, 23, 12, 6, 3, 2]`.

`SCALE_LABELS` has eleven entries: the nine scales above as `"365-day"` … `"2-day"`, then `"1-day"` for the residual component. The decomposition returns `len(scales) + 1` columns. The last column is the remainder after the finest smoother, which the paper treats as the 1-day component.

`spin_up(scales)` returns `sum(scales) - len(scales)`. For the default scales that is 723 days. Every corrected series is shorter than its input by this many leading samples. Callers that line the result up with a date axis must drop the same prefix from the dates. This helper did not exist before; the paper scripts still inline `sum(scale) - len(scale)` so their source stays close to the published scripts. New code (tests, demo, dashboard) calls `spin_up`.

Site constants, used by the dashboard and documented for the scripts: latitude −37.5, longitude 144.5, dates 1950-01-01 through 2014-12-31, WSDI base period 1961–1990, WSDI evaluation years 1952–2014. Scenarios: `historical`, `ssp126`, `ssp585`.

### 5.2 Path helpers (`src/tvc/io.py`)

`REPO_ROOT` is `Path(__file__).resolve().parents[2]`, i.e. the repository root when the file lives at `src/tvc/io.py`.

| Function | Behaviour |
|---|---|
| `data_dir()` | `$TVC_DATA_DIR` if set, otherwise `<repo>/Data` |
| `figures_dir()` | `$TVC_FIGURES_DIR` if set, otherwise `<repo>/Figures`, and `mkdir(parents=True)` |
| `data_path(name)` / `figure_path(name)` | join onto those directories |
| `load_csv_array(name, flatten=True)` | `pandas.read_csv(..., header=None)` then `numpy.array`; flattens by default |
| `load_model_names()` | model ids from column 0 of `model_name_ssp126585.csv`, display names from column 1 of `model_institution_ssp126585.csv` |
| `open_dataset(name)` | `xarray.open_dataset` of a file in `data_dir()`; imports xarray lazily |

Overriding the directories is how the paper scripts were regression-tested without touching the real `Data/` (section 7.3).

### 5.3 Plotting helper

`apply_paper_style(size)` is `matplotlib.rc("font", family="Arial", weight="normal", size=size)`. Figure scripts call it once instead of repeating the `font = {...}; matplotlib.rc("font", **font)` block. Figure 8 still builds its own `font` dict afterwards because `colorbar.set_label(..., fontdict=font)` needs the dict, not just the rcParams change.

### 5.4 Legacy shims

`src/TVC_class.py` and `src/statistics_tool.py` import the new locations and emit `DeprecationWarning` at import time (`stacklevel=2`).

```python
from TVC_class import TVC          # warns, TVC is tvc.core.TVC
from statistics_tool import statistics  # warns, same class object as tvc.statistics
```

`tests/test_compat.py` imports each shim in a fresh `sys.modules` entry and checks the warning plus object identity.

### 5.5 Proof the moved scientific code is the same

Git was unavailable. Comparison was done by parsing both files with `ast.parse` and comparing `ast.dump`:

- `TVC_class.py` (before deletion) vs `src/tvc/core.py`: identical.
- `statistics_tool.py` vs `src/tvc/metrics.py` with the final `Statistics = statistics` statement removed from the new file's AST: identical.

Ruff format was later run on the repository with `extend-exclude` covering `src/tvc/core.py`, `src/tvc/metrics.py` and `scripts/paper`, so the formatter cannot rewrite them. After formatting, a spot check confirmed `core.py` still contained the original tab-indented line `self.X_cov = self.stat_obs(X_data)`.

---

## 6. Paper scripts

Moved from the repository root into `scripts/paper/`. Analysis and plotting code is the published code. The edits in every script are:

- imports switched to `tvc`, `tvc.io`, `tvc.plotting`;
- the repeated font dict replaced by `apply_paper_style(...)` where that fully replaced it;
- `'./Data/...'` / `'./Figures/...'` replaced by `data_path(...)` / `figure_path(...)`.

`figure_path` creates `Figures/` on first use. Scripts can be run from any working directory because paths no longer depend on the current directory.

`scripts/paper/README.md` is the index. Mapping:

| New script | Was | Writes |
|---|---|---|
| `tvc_case_study.py` | `TVC_case_study.py` | nothing (prints nothing; computes variance, lag-1, lag-5, WSDI) |
| `fig01_02_S6_S7_case_study_access_esm1_5.py` | `Figure1-2_S6-7_TVC_ACCESS-ESM1-5_ERA5_case_in_sample.py` | Figure 1, Figure 2, Figure S6, Figure S7 |
| `fig03_all_models_site_metrics.py` | `Figure3_TVC_ERA5_allmodels_in_sample.py` | Figure 3, and overwrites `Data/TVC_ERA5_allmodels_-37.5_144.5_tasmax_hist.csv` |
| `fig04_05_prct_improve_boxplots.py` | `Figure4-5_box_plot_prct_improve_all_scenarios_2plots.py` | two percentage-improvement box-plot JPEGs |
| `fig06_global_aggregate_scatter.py` | `Figure6_global_aggregate_scatter_plots.py` | Figure 6 |
| `fig07_lat_aggregate_heatmap.py` | `Figure7_one_model_lat_aggregate_heatmap.py` | Figure 7 |
| `fig08_S8_global_maps.py` | `Figure8_S8_global_map_plot_6plots.py` | Figure 8, Figure S8 |
| `figS1_S4_toy_covariance_example.py` | `FigureS1-4_example_covariance_lag_correlation.py` | toy covariance heatmaps, eigenvectors, eigenvalue comparison |
| `ar1_30_500y_statistics.py` | `AR1_standard_Gaussian_nocycle_30_500years_check_statistics.py` | 18 AR(1) CSVs in `Data/` |
| `figS5_ar1_statistics_boxplots.py` | `FigureS5_Plot_30_500years_statistics.py` | Figure S5 |

Three edits go beyond import and path substitution. Each one only makes the script able to run. None changes a calculation.

1. **Missing import.** The Figure 1–2 / S6–7 script calls `timedelta` and never imported it. `from datetime import timedelta` was added. Without it the script dies while saving Figures S6 and S7, after Figures 1 and 2 have already been written.
2. **Case-study data path.** `TVC_case_study.py` read `ERA5_obs_...csv` from the current directory, not from `Data/`. It now reads through `data_path`, so it finds the files that actually ship in `Data/`.
3. **Figure S5 filename case.** The script read `obs_30y_...csv` (lowercase). The files in `Data/` are `Obs_30y_...csv`. Windows treats those as the same path; Linux does not. The script now requests `Obs_`. The comment in the original Figure S5 loop (`if c == 1: # for 30-year`) was left as written; the statistic it prints on that branch is the lag-5 error, for both the 30-year and the 500-year rows.

Figure 6, Figure 7 and Figure 8 docstrings now say which external files they need. Figure 6 and 7 cannot run from this checkout. Figure 8 cannot run until the shapefile is added.

### 6.1 Synthetic example

`examples/synthetic_demo.py` was updated to import `TVC` and `spin_up` from `tvc` and `generate_ar1` from `tvc.synthetic`. The demo procedure is the same: AR(1) observations with coefficient 0.75, model series with coefficient 0.45, scales `[90, 45, 22, 11, 5, 2]`, 4000 training and 4000 validation points, plot of the first 500 validation steps plus a scatter of corrected vs raw, saved to `outputs/tvc_synthetic_demo.png`. It was run successfully with `MPLBACKEND=Agg`.

---

## 7. Tests and verification

### 7.1 Package tests (`tests/`)

All of these passed (`12 passed`) after the refactor and again after the dashboard follow-up work.

| Test | What it locks |
|---|---|
| `test_stat_obs_returns_expected_mean_and_sample_covariance` | mean `[2, 4]` and sample covariance `[[1, 2], [2, 4]]` on a 3×2 hand-built matrix. Carried over from the original suite. |
| `test_multiscale_decomposition_shape_and_reconstruction` | component count is `len(scales) + 1`, length is `n - spin_up`, values are finite, and the components sum back to the original series after the spin-up. Carried over, now uses `spin_up`. |
| `test_in_sample_correction_reproduces_observed_mean_and_covariance` | On synthetic AR(1) series, in-sample TVC reproduces the observed component mean (atol 1e-8) and covariance (atol 1e-6). This is the property the method is built to satisfy. |
| `test_get_tvc_covariance_matches_stat_obs` | `get_TVC_covariance("train")` equals `stat_obs` on `get_decomposition("train")`. |
| `test_nday_discrete_average_returns_non_overlapping_aggregates` | 2-day means of `1..8` are `[1.5, 3.5, 5.5, 7.5]`. Carried over. |
| `test_lag1_correlation_and_variance_match_ar1_theory` | A length-40-year AR(1) with φ = 0.7 has lag-1 within 0.03 of 0.7 and variance within 0.15 of `1 / (1 - φ²)`. |
| `test_base_period_percentile_returns_one_value_per_calendar_day` | 1961–1990 daily series returns 365 finite percentiles. Guards the negative-reshape behaviour in section 4.1. |
| `test_wsdi_counts_only_spells_of_at_least_n_days` | A 7-day spell counts as 7; a 5-day spell does not count toward WSDI with `nday=6`. |
| `test_wsdi_masks_years_with_too_many_missing_days` | Five missing days in February 2002 make that year's WSDI NaN (the rule is more than 3 missing days in a month). 2001, with no gaps, is 0. |
| `test_site_case_study_metrics_are_stable` | ERA5 / ACCESS-ESM1-5 at the site. See numbers below. Skipped if the CSV is absent. |
| `test_legacy_modules_reexport_with_deprecation_warning` | Both shims warn and re-export the identical class objects. |

Site regression numbers, from a direct run of `TVC` on the shipped CSVs before the tests were written:

| Series | Variance (°C²) | Lag-1 | 5-day lag correlation |
|---|---:|---:|---:|
| ERA5, after spin-up | 41.04996 | 0.81785 | |
| ACCESS-ESM1-5 raw, after spin-up | 49.30579 | 0.87940 | |
| TVC-corrected ACCESS-ESM1-5 | 41.06988 | 0.82959 | 0.80543 |

The test checks these to 1e-3 (variance) and 1e-4 (correlations). TVC pulls variance and lag-1 from the raw model toward ERA5. That is the demonstration the dashboard's first tab is built around.

### 7.2 Dashboard tests (`app/`)

| Test | What it locks |
|---|---|
| `test_app_renders_without_exceptions` | `AppTest` runs the script, clicks "Run TVC for all models", switches the site tab to "Out-of-sample". No exception. Took about 30–33 seconds. |
| `test_box_stats_whiskers_ignore_extreme_outliers` | 99 values in `[50, 100]` plus a single −876. The lower whisker stays at or above 40. This is the Global-view bug, encoded as a test. |
| `test_box_stats_matches_matplotlib` | Quartiles and 1.5×IQR fences agree with `matplotlib.cbook.boxplot_stats` on a normal sample. |
| `test_prct_box_table_medians_are_positive_for_paper_data` | Variance, historical and ssp585: 2 × 23 rows, every median positive. |
| `test_coastlines_load` | The bundled coastline polyline has equal longitude and latitude lengths, both above 1000. |

### 7.3 Paper scripts actually executed

Run with `MPLBACKEND=Agg`, `TVC_DATA_DIR` and `TVC_FIGURES_DIR` pointed at a temp copy of `Data/` and an empty figures folder, so the real `Data/` was not modified and `Figures/` was not created in the repo.

These six completed with exit code 0 and wrote the expected JPEGs:

- `tvc_case_study.py`
- `fig01_02_S6_S7_case_study_access_esm1_5.py` (Figures 1, 2, S6, S7)
- `fig03_all_models_site_metrics.py` (Figure 3)
- `fig04_05_prct_improve_boxplots.py` (Figures 4 and 5)
- `figS1_S4_toy_covariance_example.py`
- `figS5_ar1_statistics_boxplots.py`

The Figure 3 script rewrote `TVC_ERA5_allmodels_-37.5_144.5_tasmax_hist.csv` inside the temp data copy. Compared with the file shipped in `Data/`: both are `(23018, 23)` and the maximum absolute difference is **0.0**. That is the check that the moved Figure 3 script, running against the moved `TVC` class, reproduces the published corrected series for all 23 models.

Not run:

- Figure 6 and Figure 7, because `Var_all_global_*.nc` is not in the repo.
- Figure 8 / S8, because `World_Continents.shp` is not in the repo. The NetCDFs those maps need are present; the shapefile is the missing piece. The dashboard draws the same fields without the shapefile (section 9).
- `ar1_30_500y_statistics.py`, because it is a 100-realisation random-number experiment that overwrites the `Obs_`, `Raw_` and `Corrected_` CSVs. Running it would replace published results with a new sample.

### 7.4 Lint

`ruff check .` is clean. Config: line length 110, target Python 3.9, rules `E`, `F`, `W`, `I`, `B`, `UP`. The verbatim scientific modules and `scripts/paper` are excluded. `app/` and `tests/` and `src/tvc/{constants,io,plotting,synthetic,__init__}.py` are included and were formatted with `ruff format`.

---

## 8. Dashboard

### 8.1 Why this demo, and how to run it

The recommended demo is the single-site case study, not the global maps. Reasons:

- Both series needed to *run* TVC are in the repo: `Data/ERA5_obs_-37.5_144.5_tasmax_hist.csv` and `Data/ERA5_allmodels_-37.5_144.5_tasmax_hist.csv` (23 models, shape `(23741, 23)` daily values, 1950–2014).
- One model finishes in about 0.07 seconds on this machine, so the whole 23-model panel is a few seconds.
- It exercises the actual method. The global tab only displays precomputed NetCDF fields; it does not call `TVC`.

```bash
python -m pip install -e ".[dashboard]"
streamlit run app/streamlit_app.py
```

The process used during development was `streamlit run app/streamlit_app.py --server.headless true --server.port 8599`. It was restarted once, after the Global-view fix, by killing the listener on port 8599 and starting it again. Health check: `GET http://localhost:8599/_stcore/health` returns `ok`. Address: **http://localhost:8599**.

`app/streamlit_app.py` inserts its own directory on `sys.path` and imports `dashboard_lib`. That split exists so the data functions can be tested without starting Streamlit. `dashboard_lib.py` does not import Streamlit.

### 8.2 Dependencies declared for it

`dashboard` extra in `pyproject.toml`: `streamlit>=1.45`, `plotly>=5.18`, `xarray>=2023.1`, `netCDF4>=1.6`.

The Streamlit floor was raised from a planned `>=1.30` to `>=1.45` because Streamlit 1.58 deprecates `use_container_width` (the width default is already `stretch`). Every `use_container_width=True` was removed rather than kept as a warning. `netCDF4` was added to both the `dashboard` and `research` extras because xarray 2026 refuses to open NetCDF until one of its engines is installed. `cartopy` was added to the `research` extra only, for Figure 8; the dashboard does not use it.

Installed into the local Anaconda Python while doing the work, via `pip install -e ".[dashboard,test,demo]"`: `tvc-climate 0.2.0` (editable), `netCDF4 1.7.5.1.post1`, `cftime 1.6.6.1`, `pytest-cov 7.1.0`, `coverage 7.17.3`. Separately: `ruff`, and later `kaleido` plus a Plotly-managed Chrome at `%LOCALAPPDATA%\plotly\choreographer\deps\chrome-win64\chrome.exe`. Kaleido and that Chrome are only there so Plotly figures could be written to PNG during debugging. The app does not need them.

### 8.3 Data the dashboard reads

From `dashboard_lib.load_site`:

1. Observations: `ERA5_obs_-37.5_144.5_tasmax_hist.csv`, 23741 daily values, no NaNs.
2. All models: `ERA5_allmodels_-37.5_144.5_tasmax_hist.csv`, shape `(23741, 23)`.
3. The same mean correction as Figure 3: `raw_all = raw_all + nanmean(obs) - nanmean(raw_all, axis=0)`. Each model is shifted by a single constant. The dashboard never shows the unshifted model series.
4. Display names from column 1 of `model_institution_ssp126585.csv`: ACCESS-CM2, ACCESS-ESM1-5, TaiESM1, AWI-CM-1-1-MR, BCC-CSM2-MR, FGOALS-g3, CanESM5, CNRM-CM6-1, CNRM-ESM2-1, EC-Earth3, EC-Earth3-Veg, EC-Earth3-Veg-LR, INM-CM4-8, INM-CM5-0, IPSL-CM6A-LR, MIROC6, MIROC-ES2L, MPI-ESM1-2-LR, MRI-ESM2-0, NorESM2-LM, NorESM2-MM, CNRM-CM6-1-HR, KIOST-ESM. Order matches the NetCDF `mod` coordinate on the percentage-improvement files. Checked explicitly: `list(da["mod"].values) == list(names)` is true.
5. Dates: `pd.date_range("1950-01-01", "2014-12-31")`, length 23741.

Global fields:

| Dashboard control | File pattern | Shape |
|---|---|---|
| Improvement map and box plot | `Prct_improve_{code}_MaT_mean_corr_raw_{scenario}_notrend_9_all_model_tasmax_global_0.1.nc` | `(mod=23, lat=18, lon=36)`, latitudes −89.5…80.5 step 10, longitudes 0.5…350.5 step 10 |
| Same, WSDI only | `Prct_improve_WSDI6_MaT_mean_corr_raw_new_clim_30y_2090_{scenario}_notrend_9_all_model_tasmax_global_0.1.nc` | same grid |
| TVC − raw, all-model average | `ERA5_avg_{code}_MaT_{scenario}_9_all_model_tasmax_global_test.nc` minus `ERA5_avg_{code}_MaT_{scenario}_mean_corr_raw_9_all_model_tasmax_global_test.nc` | `(lat=180, lon=360)` at 1° |
| TVC − raw, ACCESS-ESM1-5 | `{code}_ERA5_{scenario}_notrend_9_ACCESS-ESM1-5_tasmax_global_test.nc` minus the matching `mean_corr_raw` file | 1° |

`code` is `Var_30ymv`, `Lag1corr_30ymv`, `Lag5corr_30ymv`, or `WSDI6`. The difference maps have no WSDI files in `Data/`, so that dropdown excludes WSDI. Difference maps also have no `ssp126` files; the dropdown is historical and ssp585 only.

`wrap_lon` converts longitude from 0–360 to −180–180 with `((lon + 180) % 360) - 180` and sorts. That is the same conversion Figure 8 does.

Caching: `load_site`, `run_site(model_index, out_of_sample)`, `all_models_metrics`, each NetCDF field, the box-stat table and the coastline polyline are `@st.cache_data`. Changing a widget reuses the cached array instead of recomputing TVC or rereading NetCDF.

### 8.4 Tab 1 — Site case study

Default model is ACCESS-ESM1-5, the model in Figures 1 and 2.

**In-sample (the default, labelled "In-sample (paper)")** builds `TVC(obs, raw, raw, DEFAULT_SCALES)` and calls `TVC_postprocess`. The returned matrix is `(23018, 10)` = `(23741 − 723, 9 scales + residual)`. The real part is taken. Component series shown in the UI:

- observations: `get_decomposition("obs")`;
- mean-corrected raw: the training decomposition plus `Y_mean - X_mean` per component (same shift Figure 1 uses when it builds `X_corr`);
- TVC: the corrected component matrix.

Metrics are computed on the summed series with the published methods, after the spin-up:

- variance: `sample_var_30y_mv`;
- lag-1: `lag_corr_nonan_notr_30y_mv(1, 1)`;
- 5-day lag of 5-day means: `lag_corr_nonan_notr_30y_mv(5, 1)` (window 5, lag 1 on the 5-day series, which the paper calls 5-day lag correlation);
- WSDI: 90th percentile over 1961–1990 with a 5-day window, then `WSDI(6, span_yr=False, ...)`, then `nanmean` over 1952–2014.

For ACCESS-ESM1-5 in-sample the dashboard reports:

| | Variance | Lag-1 | 5-day lag | WSDI (days/year) |
|---|---:|---:|---:|---:|
| ERA5 | 41.05 | 0.818 | 0.805 | 3.206 |
| Mean-corrected raw | 49.31 | 0.879 | 0.858 | 3.556 |
| TVC | 41.07 | 0.830 | 0.805 | 2.889 |

The three non-WSDI TVC numbers are the regression-test values. The metric card's delta is the percentage improvement defined in section 8.6, labelled "closer to ERA5".

**Out-of-sample** splits at `1982-01-01`. Training observations and training model are `[:split]`. The validation model is `[split:]`. TVC is fit on 1950–1981 and applied to 1982–2014. The observation decomposition and the observation metrics on this tab are computed from the validation period only, so the reference is the same years as the corrected series. WSDI is not computed: its 1961–1990 base period is not inside 1982–2014, and `base_period_percentile` would be asked to describe a window the series does not contain. The tab shows a caption saying so.

Checked numbers for ACCESS-ESM1-5 out-of-sample:

| | Variance | Lag-1 | 5-day lag |
|---|---:|---:|---:|
| ERA5, 1982–2014 | 40.882 | 0.805 | 0.799 |
| Mean-corrected raw | 49.000 | 0.876 | 0.858 |
| TVC | 40.374 | 0.832 | 0.815 |

Charts on the tab:

- Three covariance heatmaps (ERA5, mean-corrected raw, TVC-corrected), shared colour range. The range is the max absolute entry across the three matrices, capped at 25 so one large variance does not wash the off-diagonals out. Annotations are one decimal. This is the interactive version of Figure 1, with the corrected covariance added as a third panel.
- Grouped bar chart of the diagonal (variance by scale), log y.
- Time series for one component or the sum, with a date slider. The default in-sample window starts at 1965-01-01 and runs one year, which is the year Figure 2 uses for the combined series. Out-of-sample defaults to the start of the validation output.

### 8.5 Tab 2 — Multi-model comparison

Does nothing until the button "Run TVC for all models" is pressed. The button exists because the first visit should not spend several seconds correcting 23 models before the user has asked. After the first run, `st.session_state["all_models_ran"]` keeps the panel visible across reruns, and `cached_all_models` keeps the result.

For each model it calls the same in-sample `run_site` as tab 1 and stacks a long table: model, source (`Mean-corrected raw` or `TVC-corrected`), metric, value. The ERA5 reference is computed once.

The view is a strip plot per metric (one point per model, raw group and TVC group, dashed ERA5 line). That is the interactive form of Figure 3. Under it, a table "Improvement towards ERA5 (%)" with models as rows and the four metrics as columns, coloured with a red–yellow–green gradient on −100…100.

### 8.6 Percentage improvement

Used by the site-tab cards and by the multi-model table. It is not used for the Global tab, which reads precomputed `Prct_improve_*.nc` files.

```text
improvement = ( |raw − ERA5| − |TVC − ERA5| ) / |raw − ERA5| × 100
```

Positive means TVC landed closer to ERA5 than the mean-corrected raw model did. Zero means the same distance. Negative means TVC landed further away. If the raw value equals ERA5 the denominator is zero and the result is NaN.

This measure is unstable when the raw model is already very close to ERA5, because the denominator is tiny. That is the whole of the lag-1 question in section 10. The formula itself was checked against the published series and is doing what it says.

### 8.7 Tab 3 — Global view

Two blocks.

**Percentage improvement.** Dropdowns: metric (variance, lag-1, 5-day lag, WSDI), scenario (historical, ssp126, ssp585), model (median across the 23, or one model). Map of that field, then a box plot of all three scenarios for the selected metric.

**TVC − raw difference.** Dropdowns: metric (no WSDI), scenario (historical, ssp585), source (all-model average, or ACCESS-ESM1-5). Map of the 1° difference. These are the fields in Figure 8 and Figure S8.

What the 10° files actually contain, which matters for reading the map: each latitude and longitude in the file is a single 1° cell, stored every 10°. They are not 10° area averages. The paper uses them as the sample behind the box plots in Figures 4 and 5. Drawing them as filled 10° blocks, which is what the first dashboard version did, implies a spatial average that was never computed. The replacement map keeps them as 1° cells on the coarse grid and draws a coastline on top so the cells can be located. The cells still do not represent their surrounding 10°. A note to that effect is in the tab text.

Coastlines: `app/assets/coastlines_110m.json`, 76 KB. Built from Natural Earth 1:110m coastline, which is public domain. The GeoJSON was downloaded from the `nvkelso/natural-earth-vector` mirror, coordinates were rounded to 0.01°, and the result was stored as a list of lines rather than GeoJSON. `load_coastlines` flattens the lines into one longitude array and one latitude array with `None` separators, which is how Plotly draws disjoint polylines in a single scatter trace. No cartopy, shapely or geopandas dependency. The case-study site is a star at (144.5, −37.5).

Colour scales:

- Improvement maps use `RdYlGn` (red negative, green positive) with a fixed range of −100 to +100 and a midpoint of 0.
- Difference maps use `RdBu_r` with the paper's fixed limits, not a percentile stretched to the data:

| Metric | All-model average | ACCESS-ESM1-5 |
|---|---:|---:|
| Variance | ±100 °C² | ±150 °C² |
| Lag-1 correlation | ±0.2 | ±0.3 |
| 5-day lag correlation | ±0.2 | ±0.3 |

Those limits are `DIFF_LIMITS` in `dashboard_lib.py`. They were read off `Figure8_S8_global_map_plot_6plots.py` (`levels` for `c == 0` and `c == 1`). Figure 8's all-model variance levels run −100…100 and the single-model levels run −150…150. The correlation levels are ±0.2 and ±0.3.

Box plots use the paper's definition rather than Plotly's default. `box_stats` computes the three quartiles with `numpy.percentile`, then sets the whiskers to the smallest and largest observations inside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`. Outliers are not drawn. This was checked against `matplotlib.cbook.boxplot_stats` (`whislo`, `whishi`, `med`), which is what Figures 4 and 5 use via `Axes.boxplot(..., showfliers=False)`. The y-axis is clipped to the paper's limits: −35 to 100, and −90 to 100 for WSDI (`BOX_YLIM` / `BOX_YLIM_DEFAULT`), taken from `set_ylim` in `fig04_05_prct_improve_boxplots.py`. The title says the axis is clipped so a reader does not think the whiskers are the full range of the data.

### 8.8 Tab 4 — Synthetic playground

No `Data/` access. Sliders:

- reference AR(1) coefficient, default 0.8, range 0–0.95 step 0.05;
- model AR(1) coefficient, default 0.4;
- model amplitude relative to the reference, default 1.5, range 0.5–3;
- series length, one of 2000, 4000, 8000, 16000, default 4000.

`synthetic_experiment` builds four series with `generate_ar1` and seeds `seed`, `seed+1`, `seed+2`, `seed+3` where `seed` is fixed at 7: reference training, model training, model validation, reference validation. Scales are `[90, 45, 22, 11, 5, 2]`. TVC is fit on the training pair and applied to the model validation series. The reference decomposition used for the variance bars is computed on the validation reference, so the comparison is out of sample on both sides.

Charts: variance of each component (reference, raw, TVC) on a log scale; autocorrelation to lag 30; the first 365 days of the three validation series. The autocorrelation helper is `acf` in `dashboard_lib.py`: demeaned series, `dot(x[lag:], x[:-lag]) / dot(x, x)`.

### 8.9 Global-view bug, investigation, and what changed

Reported: the grid map looked wild, and the grid-cell improvement box plot was far too negative.

Checked before changing any drawing code:

- Value distribution of every `Prct_improve_*.nc` for historical and ssp585. Variance historical is essentially all positive: 1st percentile 78.7, median 99.2, minimum 4, maximum 100. Variance ssp585 has median 69.5 and a real minimum of about −876. Lag-1 historical median is 73.7 with a minimum of −8. WSDI medians are about 30–35 with minima of a few hundred percent negative. So the files really do contain large negative outliers, and they really are a small tail. The bulk of every metric is positive.
- The same variance file, passed through the paper's own Figure 4–5 script (already run into the temp folder in section 7.3), draws boxes whose bodies sit between roughly 50 and 100 and whose whiskers only occasionally cross zero. The data and the paper agree. The dashboard's reading of model order, scenario and variable name was not the problem. `prct_improve("Variance", "ssp585")` medians per model are all positive (ACCESS-CM2 68, ACCESS-ESM1-5 70, and so on).
- Plotly's `px.box` on the raw cell values, rendered to PNG with Kaleido, stretched the y-axis to about −800. Plotly's default whisker runs to the sample minimum and maximum. Matplotlib's default, which the paper uses, stops at 1.5×IQR and hides the rest as fliers, and the paper additionally turns fliers off and clips the axis to −35…100. Same numbers, different box definition. That was the whole of the "too negative" complaint.
- `px.imshow` of the 18×36 improvement field, rendered the same way, produced a banded rectangle with no coastline and cells the size of countries. Combined with a colour range of −100…100 on a field whose historical values sit near 99, it reads as a broken grid. The 1° difference map drawn the same way was recognisable as a world map but its colour range was `nanpercentile(|diff|, 98)`, which for historical variance was ±52. The paper uses ±100. The percentile choice compressed the scale and was not comparable to Figure 8.

Changes, all in `app/dashboard_lib.py` and `app/streamlit_app.py`:

- `box_stats` and `prct_box_table`. The box plot is now a `go.Box` built from precomputed quartiles and fences (`q1`, `median`, `q3`, `lowerfence`, `upperfence`, `mean`), one trace per scenario, `boxmode="group"`.
- `world_map` replaced `px.imshow` with `go.Heatmap` (`zsmooth=False`), an equirectangular axis (`scaleanchor="x"`, `scaleratio=1`, longitude −180…180, latitude −90…90), the coastline polyline, and the site star.
- Difference-map colour limits switched to `DIFF_LIMITS`.
- Tests in section 7.2 added.
- The Global tab copy now says the cells are sampled every 10°, that positive means TVC is closer to ERA5, and that the box axis is clipped to the paper's range.

Re-rendered from the running app via Streamlit `AppTest` (so the figures are exactly what the script produces, not a copy of the plotting code):

- Historical variance, median across models: green at high latitudes and mid-latitudes, a lighter green band in the tropics, coastlines and the site star in place. Matches the historical column of Figure 4, where almost every box is up against 100.
- Variance box plot: historical boxes near 100, ssp126 and ssp585 bodies roughly 50–90, a few ssp585 whiskers crossing zero (CanESM5 to about −15, BCC-CSM2-MR to about −5). Compared visually with the JPEG produced by `fig04_05_prct_improve_boxplots.py`. Same models cross zero, by similar amounts.
- Historical all-model variance difference: the 1° field with ±100 limits. Land and ocean structure is visible; high-latitude land is the strongest colour.

Historical improvement maps looking almost uniformly green is what the file contains (median 99.2). It is the in-sample fit. It is not a colour-scale bug. ssp585 is the scenario that shows spatial structure on that map.

The first dashboard version's box plot and maps were wrong as pictures. The numbers underneath them were the published numbers.

### 8.10 Lag-1 negatives in the multi-model table

Reported: three negative cells in "Improvement towards ERA5 (%)" for lag-1 correlation.

Recomputed from scratch, not from the dashboard table. For each of the 23 models:

- raw lag-1 from the mean-corrected `ERA5_allmodels_...csv`, after the 723-day spin-up, through `statistics.lag_corr_nonan_notr_30y_mv(1, 1)`;
- TVC lag-1 from `run_site` (the dashboard path);
- TVC lag-1 again from the shipped `TVC_ERA5_allmodels_-37.5_144.5_tasmax_hist.csv`, which Figure 3 wrote when the paper was produced;
- improvement with the formula in section 8.6, against ERA5 lag-1 of 0.817847.

The dashboard TVC lag-1 and the published-file lag-1 differ by at most about 5×10⁻⁹ for every model. The three negative improvements are properties of the published corrected series, not of the dashboard.

ERA5 lag-1 at this site: **0.81785**.

| Model | Raw lag-1 | TVC lag-1 | Raw distance to ERA5 | TVC distance to ERA5 | Improvement |
|---|---:|---:|---:|---:|---:|
| CNRM-CM6-1-HR | 0.81837 | 0.82090 | 0.00052 | 0.00305 | −480% |
| MRI-ESM2-0 | 0.81003 | 0.80545 | 0.00782 | 0.01240 | −59% |
| FGOALS-g3 | 0.82398 | 0.81065 | 0.00613 | 0.00720 | −17% |

The other 20 models improve, from 68% (CNRM-CM6-1) to 96% (CanESM5).

CNRM-CM6-1-HR is the misleading one. Its raw lag-1 was already 0.0005 from ERA5. TVC moved it to 0.821, which is still close in absolute terms, but the raw error in the denominator is so small that the percentage is −480. MRI-ESM2-0 was slightly low and TVC pushed it lower. FGOALS-g3 was slightly high and TVC overshot to slightly low, by a similar distance, so the percentage is mildly negative.

Why a method that matches covariance can move lag-1 the wrong way for a few models: TVC matches the mean and covariance of the ten scale components. Lag-1 of the summed series is a different statistic. A model whose lag-1 happened to be right can be moved a little when the component covariance is corrected. The site tab for ACCESS-ESM1-5 shows the typical case, where both variance and lag-1 move toward ERA5 (49.3 → 41.1 against an ERA5 variance of 41.0; lag-1 0.879 → 0.830 against 0.818).

No code change followed this check.

---

## 9. Tooling and CI

`.github/workflows/tests.yml` has three jobs:

| Job | What it runs |
|---|---|
| `lint` | `ruff check .` on Python 3.12 |
| `test` | `pip install -e ".[test]"`, then `pytest --cov=tvc --cov-report=term-missing`, on Python 3.9, 3.10, 3.11, 3.12 |
| `smoke` | `pip install -e ".[demo,dashboard]"`, `MPLBACKEND=Agg python examples/synthetic_demo.py`, and `pytest app/test_app.py` |

The original workflow was the `test` job only, on 3.10–3.12, without coverage, and it installed `.[test]`.

`.pre-commit-config.yaml`: `end-of-file-fixer`, `trailing-whitespace` (excluded on the verbatim files), `check-yaml`, `check-added-large-files` with a 6 MB limit, `ruff` with `--fix`, `ruff-format`. Pre-commit was not executed locally; the hooks are the same tools CI runs.

`.gitignore` gained `Figures/`, `.mypy_cache/`, `.ruff_cache/`, `.streamlit/secrets.toml`. `outputs/` was already ignored.

`[tool.mypy]` is configured (Python 3.9, `ignore_missing_imports`, paper scripts excluded) but mypy was not run as part of this work and is not a CI step.

Docs written or replaced:

| File | Content |
|---|---|
| `README.md` | layout, extras table, dashboard section, new import example, pointer at `scripts/paper/README.md`, development commands |
| `scripts/paper/README.md` | script-to-figure table, old filenames, missing inputs |
| `CHANGELOG.md` | 0.2.0 structural release, 0.1.0 initial packaged release |
| `CONTRIBUTING.md` | install, keep `core.py` and `metrics.py` numerically identical unless a change is deliberate, commands to run before a pull request |
| `REFACTOR.md` | this file |

`CITATION.cff` was not changed.

---

## 10. Public API after the refactor

```python
from tvc import TVC, DEFAULT_SCALES, spin_up, statistics

model = TVC(obs, raw_train, raw_val, DEFAULT_SCALES)
components = model.TVC_postprocess()          # may be complex; take np.real if needed
corrected = components.sum(axis=1)            # aligns with dates[spin_up(DEFAULT_SCALES):]
```

`statistics` is still the class name used everywhere in the paper scripts. `Statistics` is an alias for new code. Old imports keep working:

```python
from TVC_class import TVC                 # DeprecationWarning
from statistics_tool import statistics    # DeprecationWarning
```

---

## 11. Known limits, left as they are

- **Method code is frozen on purpose.** The items in section 4 stay, including `exit()` on a length mismatch and the hard-coded 10957-day window.
- **Figure 6 and Figure 7 cannot be regenerated** from this checkout. They need `Var_all_global_{historical,ssp585}_tasmax_{raw,TVC}.nc`.
- **Figure 8 cannot be regenerated** until `Data/World_Continents.shp` (and its sidecar files) is downloaded. The dashboard's difference maps are the same NetCDF fields without the shapefile.
- **`ar1_30_500y_statistics.py` uses unseeded `numpy.random`.** A rerun will not reproduce the CSVs in `Data/`. It was not run.
- **Historical global improvement is ~99% almost everywhere.** That is the in-sample fit stored in the NetCDF, and it matches Figure 4. The map is not broken when it looks flat and green.
- **The 10° improvement grid is a subsample of 1° cells.** The map does not claim those cells are 10° means. Box plots are the summary the paper publishes for that sample.
- **Percentage improvement explodes when the raw error is ~0.** CNRM-CM6-1-HR lag-1 at −480% is the example. The absolute lag-1 values are the right thing to read in that row.
- **Out-of-sample mode has no WSDI**, because the 1961–1990 base period is outside the 1982–2014 validation window.
- **Default `pytest` does not collect `app/`.** Use `pytest app` for the dashboard tests. CI's smoke job runs `app/test_app.py` only; `app/test_dashboard_lib.py` runs when the app folder is collected as a whole.
- **mypy is configured and not wired into CI.**
- **Arial** is what `apply_paper_style` requests. Machines without Arial fall back to Matplotlib's default. The scripts behaved that way before the move too.
- **Kaleido and the Plotly Chrome build** are installed in the local environment and are not project dependencies.

---

## 12. Commands used to confirm the result

```bash
python -m pip install -e ".[dashboard,test,demo]"
python -m ruff check .
python -m pytest                 # 12 passed
python -m pytest app             # 5 passed
MPLBACKEND=Agg python examples/synthetic_demo.py
streamlit run app/streamlit_app.py --server.headless true --server.port 8599
```

Paper scripts, against a temp copy of `Data/` (PowerShell):

```powershell
$tmp = Join-Path $env:TEMP "tvc_check"
New-Item -ItemType Directory $tmp
Copy-Item Data "$tmp\Data" -Recurse
$env:TVC_DATA_DIR = "$tmp\Data"
$env:TVC_FIGURES_DIR = "$tmp\Figures"
$env:MPLBACKEND = "Agg"
python scripts/paper/fig03_all_models_site_metrics.py
```

Figure 3's regenerated CSV matched `Data/TVC_ERA5_allmodels_-37.5_144.5_tasmax_hist.csv` with maximum absolute difference 0.
