# Contributing

1. Install the development environment: `python -m pip install -e ".[dev]"` and `pre-commit install`.
2. Keep `src/tvc/core.py` and `src/tvc/metrics.py` numerically identical to the published method unless
   a change is intended and documented in `CHANGELOG.md`; `tests/test_site_regression.py` guards the
   case-study results.
3. Scripts in `scripts/paper/` reproduce published figures; change their analysis only to fix errors.
4. Before opening a pull request run `ruff check .`, `python -m pytest` and `python -m pytest app`.
