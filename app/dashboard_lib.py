"""Data preparation for the Streamlit dashboard (no Streamlit imports, so it is testable)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tvc import DEFAULT_SCALES, TVC, spin_up, statistics
from tvc.constants import (
    SITE_END,
    SITE_START,
    WSDI_BASE_END,
    WSDI_BASE_START,
    WSDI_YEAR_END,
    WSDI_YEAR_START,
)
from tvc.io import load_csv_array, load_model_names, open_dataset
from tvc.synthetic import generate_ar1

OBS_FILE = "ERA5_obs_-37.5_144.5_tasmax_hist.csv"
ALL_MODELS_FILE = "ERA5_allmodels_-37.5_144.5_tasmax_hist.csv"

METRICS = {
    "Variance": "Var_30ymv",
    "Lag-1 correlation": "Lag1corr_30ymv",
    "5-day lag correlation": "Lag5corr_30ymv",
    "WSDI": "WSDI6",
}
METRIC_UNITS = {"Variance": "°C²", "Lag-1 correlation": "", "5-day lag correlation": "", "WSDI": "days/yr"}

OUT_OF_SAMPLE_SPLIT = "1982-01-01"


def load_site():
    """Observations, mean-corrected raw series for every model, model names and dates (as in Figure 3)."""
    obs = load_csv_array(OBS_FILE)
    raw_all = load_csv_array(ALL_MODELS_FILE, flatten=False)
    raw_all = raw_all + np.nanmean(obs) - np.nanmean(raw_all, axis=0)
    _, names = load_model_names()
    dates = pd.date_range(SITE_START, SITE_END)
    return obs, raw_all, [str(n) for n in names], dates


def site_metrics(series, dates, with_wsdi=True):
    stats = statistics(series)
    out = {
        "Variance": stats.sample_var_30y_mv(),
        "Lag-1 correlation": stats.lag_corr_nonan_notr_30y_mv(1, 1),
        "5-day lag correlation": stats.lag_corr_nonan_notr_30y_mv(5, 1),
    }
    if with_wsdi:
        in_base = (dates.year >= WSDI_BASE_START) & (dates.year <= WSDI_BASE_END)
        prcts = stats.base_period_percentile(series[in_base], WSDI_BASE_START, WSDI_BASE_END, 5)
        out["WSDI"] = float(np.nanmean(stats.WSDI(6, False, prcts, WSDI_YEAR_START, WSDI_YEAR_END)))
    return out


def run_site(obs, raw, dates, out_of_sample=False, scales=DEFAULT_SCALES):
    """Run TVC for one model.

    In-sample reproduces the paper (fit and apply over 1950-2014). Out-of-sample fits on
    1950-1981 and applies the correction to 1982-2014; WSDI is skipped there because the
    1961-1990 percentile base period is not inside the validation window.
    """
    init = spin_up(scales)
    if out_of_sample:
        split = dates.get_loc(pd.Timestamp(OUT_OF_SAMPLE_SPLIT))
        model = TVC(obs[:split], raw[:split], raw[split:], scales)
        corrected = np.real(model.TVC_postprocess())
        obs_eval, raw_eval = obs[split:], raw[split:]
        obs_components = TVC(obs_eval, obs_eval, obs_eval, scales).get_decomposition("obs")
        raw_components = model.get_decomposition("val")
        eval_dates = dates[split:]
    else:
        model = TVC(obs, raw, raw, scales)
        corrected = np.real(model.TVC_postprocess())
        obs_eval, raw_eval = obs, raw
        obs_components = model.get_decomposition("obs")
        raw_components = model.get_decomposition("train")
        eval_dates = dates

    raw_mc = raw_components + model.Y_mean - model.X_mean
    out_dates = eval_dates[init:]
    with_wsdi = not out_of_sample
    return {
        "dates": out_dates,
        "obs": obs_components,
        "raw": raw_mc,
        "tvc": corrected,
        "cov": {
            "ERA5": model.stat_obs(obs_components)[1],
            "Mean-corrected raw": model.stat_obs(raw_components)[1],
            "TVC-corrected": model.stat_obs(corrected)[1],
        },
        "metrics": {
            "ERA5": site_metrics(obs_eval[init:], out_dates, with_wsdi),
            "Mean-corrected raw": site_metrics(raw_eval[init:], out_dates, with_wsdi),
            "TVC-corrected": site_metrics(corrected.sum(axis=1), out_dates, with_wsdi),
        },
    }


def percent_improvement(obs_value, raw_value, tvc_value):
    raw_err = abs(raw_value - obs_value)
    if raw_err == 0:
        return float("nan")
    return (raw_err - abs(tvc_value - obs_value)) / raw_err * 100


def all_models_metrics(obs, raw_all, names, dates, progress=None):
    """Long-form table of in-sample metrics for every model (raw vs TVC) plus ERA5 reference values."""
    init = spin_up(DEFAULT_SCALES)
    obs_metrics = site_metrics(obs[init:], dates[init:])
    rows = []
    for m, name in enumerate(names):
        result = run_site(obs, raw_all[:, m], dates)
        for source in ("Mean-corrected raw", "TVC-corrected"):
            for metric, value in result["metrics"][source].items():
                rows.append({"model": name, "source": source, "metric": metric, "value": value})
        if progress is not None:
            progress((m + 1) / len(names), name)
    return pd.DataFrame(rows), obs_metrics


def wrap_lon(da):
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")


def prct_improve(metric, scenario):
    code = METRICS[metric]
    period = f"new_clim_30y_2090_{scenario}" if code == "WSDI6" else scenario
    name = f"Prct_improve_{code}_MaT_mean_corr_raw_{period}_notrend_9_all_model_tasmax_global_0.1.nc"
    return wrap_lon(open_dataset(name)["tasmax"].load())


def tvc_minus_raw(metric, scenario, source):
    """1-degree TVC minus mean-corrected raw difference maps used in Figure 8 / S8."""
    code = METRICS[metric]
    if source == "All-model average":
        tvc = open_dataset(f"ERA5_avg_{code}_MaT_{scenario}_9_all_model_tasmax_global_test.nc")
        raw = open_dataset(f"ERA5_avg_{code}_MaT_{scenario}_mean_corr_raw_9_all_model_tasmax_global_test.nc")
    else:
        tvc = open_dataset(f"{code}_ERA5_{scenario}_notrend_9_{source}_tasmax_global_test.nc")
        raw = open_dataset(f"{code}_ERA5_{scenario}_notrend_9_{source}_mean_corr_raw_tasmax_test.nc")
    return wrap_lon((tvc - raw)["tasmax"].load())


COASTLINES_FILE = Path(__file__).with_name("assets") / "coastlines_110m.json"

# Colour limits and y-axis limits taken from the paper figures (Figures 4-5 and 8 / S8).
DIFF_LIMITS = {
    ("Variance", "All-model average"): 100.0,
    ("Variance", "ACCESS-ESM1-5"): 150.0,
    ("Lag-1 correlation", "All-model average"): 0.2,
    ("Lag-1 correlation", "ACCESS-ESM1-5"): 0.3,
    ("5-day lag correlation", "All-model average"): 0.2,
    ("5-day lag correlation", "ACCESS-ESM1-5"): 0.3,
}
BOX_YLIM = {"WSDI": (-90, 100)}
BOX_YLIM_DEFAULT = (-35, 100)


def load_coastlines():
    """Natural Earth 110m coastlines (public domain) as one NaN-separated polyline for Plotly."""
    lons, lats = [], []
    for line in json.loads(COASTLINES_FILE.read_text()):
        lons += [p[0] for p in line] + [None]
        lats += [p[1] for p in line] + [None]
    return lons, lats


def box_stats(values):
    """Quartiles with 1.5 x IQR whiskers clipped to the data, as in the paper's Matplotlib box plots."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[~np.isnan(values)]
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    return {
        "q1": q1,
        "median": med,
        "q3": q3,
        "lowerfence": values[values >= q1 - 1.5 * iqr].min(),
        "upperfence": values[values <= q3 + 1.5 * iqr].max(),
        "mean": values.mean(),
    }


def prct_box_table(metric, scenarios):
    """One row of box statistics per (model, scenario) for the grid-cell improvement."""
    rows = []
    for scenario in scenarios:
        da = prct_improve(metric, scenario)
        for model in da["mod"].values:
            rows.append({"model": str(model), "scenario": scenario, **box_stats(da.sel(mod=model).values)})
    return pd.DataFrame(rows)


def acf(x, max_lag):
    x = np.asarray(x) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[lag:], x[:-lag]) / denom for lag in range(1, max_lag + 1)])


def synthetic_experiment(phi_obs, phi_model, model_sd, length, seed, scales):
    """Fit TVC on one synthetic pair and apply it to independent model and reference series."""
    obs_train = generate_ar1(length, phi_obs, seed)
    raw_train = model_sd * generate_ar1(length, phi_model, seed + 1)
    raw_val = model_sd * generate_ar1(length, phi_model, seed + 2)
    obs_val = generate_ar1(length, phi_obs, seed + 3)

    model = TVC(obs_train, raw_train, raw_val, scales)
    corrected = np.real(model.TVC_postprocess())
    obs_components = TVC(obs_val, obs_val, obs_val, scales).get_decomposition("obs")
    raw_components = model.get_decomposition("val")
    init = spin_up(scales)
    return {
        "obs": obs_val[init:],
        "raw": raw_val[init:],
        "tvc": corrected.sum(axis=1),
        "var": {
            "Reference": np.var(obs_components, axis=0, ddof=1),
            "Raw model": np.var(raw_components, axis=0, ddof=1),
            "TVC-corrected": np.var(corrected, axis=0, ddof=1),
        },
    }
