# Changelog

## 0.2.0

Structural refactor. The TVC and metric algorithms are unchanged.

### Added
- `tvc` package under `src/` (`core`, `metrics`, `constants`, `io`, `plotting`, `synthetic`).
- Streamlit + Plotly demo dashboard in `app/`.
- Tests for in-sample covariance matching, AR(1) lag correlation and variance, WSDI spell counting and
  missing-data masking, the site case-study regression, and the legacy import shims.
- Ruff, mypy and pre-commit configuration; CI lint and dashboard smoke jobs; Python 3.9 in the CI matrix.

### Changed
- Paper scripts moved to `scripts/paper/` with descriptive names; they read `Data/` and write `Figures/`
  via `tvc.io`, so they can be run from any working directory.
- `TVC_class` and `statistics_tool` are now deprecated re-export shims for `tvc`.

### Fixed
- Figure 1-2 / S6-7 script: missing `timedelta` import.
- Case-study script: input CSVs are read from `Data/`.
- Figure S5 script: `Obs_` file prefix now matches the files in `Data/` on case-sensitive file systems.

## 0.1.0

Initial packaged release of the code accompanying Shao et al. (2024).
