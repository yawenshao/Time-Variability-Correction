"""Interactive demo of Time Variability Correction (TVC).

Run with:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dashboard_lib as lib  # noqa: E402

from tvc import SCALE_LABELS  # noqa: E402
from tvc.constants import SCENARIOS, SITE_LAT, SITE_LON  # noqa: E402

COLORS = {"ERA5": "#d62728", "Mean-corrected raw": "#7f7f7f", "TVC-corrected": "#1f77b4"}
SYNTH_COLORS = {"Reference": "#d62728", "Raw model": "#7f7f7f", "TVC-corrected": "#1f77b4"}

st.set_page_config(page_title="Time Variability Correction demo", layout="wide")


@st.cache_data(show_spinner=False)
def cached_site():
    return lib.load_site()


@st.cache_data(show_spinner="Running TVC...")
def cached_run(model_index, out_of_sample):
    obs, raw_all, _, dates = cached_site()
    return lib.run_site(obs, raw_all[:, model_index], dates, out_of_sample)


@st.cache_data(show_spinner=False)
def cached_all_models():
    obs, raw_all, names, dates = cached_site()
    return lib.all_models_metrics(obs, raw_all, names, dates)


@st.cache_data(show_spinner=False)
def cached_prct(metric, scenario):
    return lib.prct_improve(metric, scenario)


@st.cache_data(show_spinner=False)
def cached_prct_boxes(metric):
    return lib.prct_box_table(metric, SCENARIOS)


@st.cache_data(show_spinner=False)
def cached_coastlines():
    return lib.load_coastlines()


@st.cache_data(show_spinner=False)
def cached_diff(metric, scenario, source):
    return lib.tvc_minus_raw(metric, scenario, source)


def fmt(metric, value):
    return f"{value:.2f}" if metric in ("Variance", "WSDI") else f"{value:.3f}"


def covariance_heatmap(cov, title, zmax):
    fig = px.imshow(
        cov,
        x=SCALE_LABELS,
        y=SCALE_LABELS,
        color_continuous_scale="RdBu_r",
        zmin=-zmax,
        zmax=zmax,
        text_auto=".1f",
        aspect="equal",
    )
    fig.update_layout(title=title, height=430, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
    fig.update_traces(textfont_size=9)
    return fig


def world_map(da, title, colorscale, zmid=0.0, zrange=None, colorbar_title=""):
    """Equirectangular map of a lon/lat field with coastlines drawn on top."""
    if zrange is None:
        zrange = (None, None)
    fig = go.Figure(
        go.Heatmap(
            z=da.values,
            x=da.lon.values,
            y=da.lat.values,
            colorscale=colorscale,
            zmin=zrange[0],
            zmax=zrange[1],
            zmid=zmid,
            zsmooth=False,
            colorbar=dict(title=colorbar_title, thickness=14),
            hovertemplate="lon %{x}, lat %{y}<br>%{z:.3g}<extra></extra>",
        )
    )
    coast_lon, coast_lat = cached_coastlines()
    fig.add_scatter(
        x=coast_lon,
        y=coast_lat,
        mode="lines",
        line=dict(width=0.6, color="rgba(30,30,30,0.8)"),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scatter(
        x=[SITE_LON],
        y=[SITE_LAT],
        mode="markers",
        marker=dict(symbol="star", size=11, color="black", line=dict(width=1, color="white")),
        showlegend=False,
        hovertemplate="Case-study site<extra></extra>",
    )
    fig.update_xaxes(range=[-180, 180], title="Longitude", constrain="domain")
    fig.update_yaxes(range=[-90, 90], title="Latitude", scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(title=title, height=460, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def improvement_box_plot(metric):
    """Grid-cell improvement by model, boxes as in the paper (1.5 x IQR whiskers, no outliers)."""
    table = cached_prct_boxes(metric)
    fig = go.Figure()
    for scenario, color in zip(SCENARIOS, ("#2ca02c", "#ff7f0e", "#d62728")):
        sub = table[table.scenario == scenario]
        fig.add_trace(
            go.Box(
                name=scenario,
                x=sub["model"],
                q1=sub["q1"],
                median=sub["median"],
                q3=sub["q3"],
                lowerfence=sub["lowerfence"],
                upperfence=sub["upperfence"],
                mean=sub["mean"],
                marker_color=color,
                offsetgroup=scenario,
            )
        )
    ylim = lib.BOX_YLIM.get(metric, lib.BOX_YLIM_DEFAULT)
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_yaxes(range=list(ylim), title="Improvement (%)")
    fig.update_layout(
        boxmode="group",
        title=f"{metric}: grid-cell improvement by model (axis clipped to the paper's range)",
        height=480,
        margin=dict(l=60, r=10, t=80, b=10),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
    )
    fig.update_xaxes(tickangle=-60)
    return fig


st.title("Time Variability Correction (TVC)")
st.caption(
    "Shao et al. (2024), *Time variability correction of CMIP6 climate change projections*, JAMES. "
    "[doi:10.1029/2023MS003640](https://doi.org/10.1029/2023MS003640) · daily maximum temperature (tasmax)."
)

obs, raw_all, model_names, dates = cached_site()

site_tab, models_tab, global_tab, synth_tab = st.tabs(
    ["Site case study", "Multi-model comparison", "Global view", "Synthetic playground"]
)

# --------------------------------------------------------------------------- Site case study
with site_tab:
    st.markdown(
        f"ERA5 versus CMIP6 historical tasmax at **{SITE_LAT}°, {SITE_LON}°** (near Melbourne), 1950–2014. "
        "TVC splits each series into time-scale components and matches the model's component "
        "mean and covariance to the observations."
    )
    c1, c2 = st.columns([2, 1])
    model_name = c1.selectbox("CMIP6 model", model_names, index=model_names.index("ACCESS-ESM1-5"))
    mode = c2.radio(
        "Evaluation",
        ["In-sample (paper)", "Out-of-sample"],
        horizontal=True,
        help="Out-of-sample fits TVC on 1950–1981 and applies it to 1982–2014.",
    )
    result = cached_run(model_names.index(model_name), mode == "Out-of-sample")

    metric_cols = st.columns(len(result["metrics"]["ERA5"]))
    for col, metric in zip(metric_cols, result["metrics"]["ERA5"]):
        o = result["metrics"]["ERA5"][metric]
        r = result["metrics"]["Mean-corrected raw"][metric]
        t = result["metrics"]["TVC-corrected"][metric]
        col.metric(
            f"{metric} {lib.METRIC_UNITS[metric]}".strip(),
            fmt(metric, t),
            delta=f"{lib.percent_improvement(o, r, t):.0f}% closer to ERA5",
            help=f"ERA5 {fmt(metric, o)} · raw {fmt(metric, r)} · TVC {fmt(metric, t)}",
        )
        col.caption(f"ERA5 {fmt(metric, o)} · raw {fmt(metric, r)}")
    if mode == "Out-of-sample":
        st.caption(
            "WSDI is in-sample only: its 1961–1990 percentile base period lies outside the validation window."
        )

    st.subheader("Covariance between time-scale components")
    zmax = float(max(np.abs(c).max() for c in result["cov"].values()))
    zmax = min(zmax, 25.0)
    cov_cols = st.columns(3)
    for col, (label, cov) in zip(cov_cols, result["cov"].items()):
        col.plotly_chart(covariance_heatmap(cov, label, zmax))

    st.subheader("Variance by time scale")
    var_df = (
        pd.DataFrame({label: np.diag(cov) for label, cov in result["cov"].items()}, index=SCALE_LABELS)
        .reset_index(names="scale")
        .melt(id_vars="scale", var_name="series", value_name="variance")
    )
    fig = px.bar(
        var_df,
        x="scale",
        y="variance",
        color="series",
        barmode="group",
        log_y=True,
        color_discrete_map=COLORS,
        labels={"variance": "Variance (°C², log scale)"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend_title=None)
    st.plotly_chart(fig)

    st.subheader("Time series")
    s1, s2 = st.columns([1, 2])
    component = s1.selectbox("Component", ["Combined", *SCALE_LABELS], index=0)
    out_dates = result["dates"]
    default_start = max(out_dates[0], pd.Timestamp("1965-01-01")) if mode.startswith("In") else out_dates[0]
    start, end = s2.slider(
        "Date range",
        min_value=out_dates[0].to_pydatetime(),
        max_value=out_dates[-1].to_pydatetime(),
        value=(default_start.to_pydatetime(), (default_start + pd.DateOffset(years=1)).to_pydatetime()),
        format="YYYY-MM-DD",
    )
    window = (out_dates >= pd.Timestamp(start)) & (out_dates <= pd.Timestamp(end))
    fig = go.Figure()
    for label, key in (("ERA5", "obs"), ("Mean-corrected raw", "raw"), ("TVC-corrected", "tvc")):
        values = (
            result[key].sum(axis=1)
            if component == "Combined"
            else result[key][:, SCALE_LABELS.index(component)]
        )
        fig.add_scatter(
            x=out_dates[window],
            y=values[window],
            name=label,
            mode="lines",
            line=dict(width=1.2, color=COLORS[label]),
        )
    fig.update_layout(
        height=380,
        yaxis_title="Temperature (°C)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig)

# --------------------------------------------------------------------------- Multi-model comparison
with models_tab:
    st.markdown(
        "In-sample metrics for all 23 CMIP6 models at the case-study site (interactive Figure 3). "
        "The dashed line is the ERA5 value."
    )
    if st.button("Run TVC for all models", type="primary") or st.session_state.get("all_models_ran"):
        st.session_state["all_models_ran"] = True
        with st.spinner("Correcting 23 models and computing metrics..."):
            table, obs_metrics = cached_all_models()

        metric_cols = st.columns(2)
        for i, metric in enumerate(obs_metrics):
            sub = table[table.metric == metric]
            fig = px.strip(
                sub,
                x="source",
                y="value",
                color="model",
                hover_name="model",
                labels={"value": f"{metric} {lib.METRIC_UNITS[metric]}".strip(), "source": ""},
            )
            fig.add_hline(
                y=obs_metrics[metric],
                line_dash="dash",
                line_color="black",
                annotation_text="ERA5",
                annotation_position="top left",
            )
            fig.update_traces(marker_size=9, jitter=0.5)
            fig.update_layout(title=metric, showlegend=False, height=360, margin=dict(l=10, r=10, t=40, b=10))
            metric_cols[i % 2].plotly_chart(fig)

        wide = table.pivot_table(index=["model", "metric"], columns="source", values="value").reset_index()
        wide["improvement %"] = [
            lib.percent_improvement(obs_metrics[m], r, t)
            for m, r, t in zip(wide.metric, wide["Mean-corrected raw"], wide["TVC-corrected"])
        ]
        summary = wide.pivot(index="model", columns="metric", values="improvement %")[list(obs_metrics)]
        st.subheader("Improvement towards ERA5 (%)")
        st.dataframe(summary.style.format("{:.0f}").background_gradient(cmap="RdYlGn", vmin=-100, vmax=100))
    else:
        st.info("Click the button to correct all 23 models; this takes a few seconds the first time.")

# --------------------------------------------------------------------------- Global view
with global_tab:
    st.markdown(
        "Precomputed global results from the paper: percentage improvement of TVC over the mean-corrected "
        "raw model at grid cells sampled every 10° (Figures 4–5; positive = TVC closer to ERA5), "
        "and 1° TVC − raw difference maps with the paper's colour ranges (Figure 8 / S8). "
        "The star marks the case-study site."
    )
    g1, g2, g3 = st.columns(3)
    metric = g1.selectbox("Metric", list(lib.METRICS), key="global_metric")
    scenario = g2.selectbox("Scenario", SCENARIOS, key="global_scenario")
    prct = cached_prct(metric, scenario)
    model_choice = g3.selectbox("Model", ["Median across models", *map(str, prct["mod"].values)])

    field = prct.median("mod") if model_choice == "Median across models" else prct.sel(mod=model_choice)
    st.plotly_chart(
        world_map(
            field,
            f"{metric} improvement, {scenario} · {model_choice}",
            "RdYlGn",
            zmid=0,
            zrange=(-100, 100),
            colorbar_title="%",
        ),
    )

    st.plotly_chart(improvement_box_plot(metric))

    st.subheader("TVC − raw difference (1°)")
    d1, d2, d3 = st.columns(3)
    diff_metric = d1.selectbox("Metric", [m for m in lib.METRICS if m != "WSDI"], key="diff_metric")
    diff_scenario = d2.selectbox("Scenario", ["historical", "ssp585"], key="diff_scenario")
    diff_source = d3.selectbox("Source", ["All-model average", "ACCESS-ESM1-5"], key="diff_source")
    diff = cached_diff(diff_metric, diff_scenario, diff_source)
    limit = lib.DIFF_LIMITS[(diff_metric, diff_source)]
    st.plotly_chart(
        world_map(
            diff,
            f"{diff_metric}: TVC − raw, {diff_scenario} · {diff_source}",
            "RdBu_r",
            zmid=0,
            zrange=(-limit, limit),
            colorbar_title="Δ",
        ),
    )

# --------------------------------------------------------------------------- Synthetic playground
with synth_tab:
    st.markdown(
        "Generate a synthetic reference and a mis-specified model as AR(1) processes, fit TVC on one "
        "realisation and apply it to an independent one. No external data are used."
    )
    p1, p2, p3, p4 = st.columns(4)
    phi_obs = p1.slider("Reference AR(1) coefficient", 0.0, 0.95, 0.8, 0.05)
    phi_model = p2.slider("Model AR(1) coefficient", 0.0, 0.95, 0.4, 0.05)
    model_sd = p3.slider("Model amplitude (× reference)", 0.5, 3.0, 1.5, 0.1)
    length = p4.select_slider("Series length (days)", [2000, 4000, 8000, 16000], value=4000)
    synth_scales = [90, 45, 22, 11, 5, 2]
    synth = lib.synthetic_experiment(phi_obs, phi_model, model_sd, length, 7, synth_scales)
    synth_labels = [f"{s}-day" for s in synth_scales] + ["residual"]

    a, b = st.columns(2)
    var_df = (
        pd.DataFrame(synth["var"], index=synth_labels)
        .reset_index(names="scale")
        .melt(id_vars="scale", var_name="series", value_name="variance")
    )
    fig = px.bar(
        var_df,
        x="scale",
        y="variance",
        color="series",
        barmode="group",
        log_y=True,
        color_discrete_map=SYNTH_COLORS,
        title="Variance by time scale (validation)",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend_title=None)
    a.plotly_chart(fig)

    lags = np.arange(1, 31)
    fig = go.Figure()
    for label, key in (("Reference", "obs"), ("Raw model", "raw"), ("TVC-corrected", "tvc")):
        fig.add_scatter(
            x=lags,
            y=lib.acf(synth[key], 30),
            name=label,
            mode="lines+markers",
            line=dict(color=SYNTH_COLORS[label]),
            marker_size=4,
        )
    fig.update_layout(
        title="Autocorrelation (validation)",
        xaxis_title="Lag (days)",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    b.plotly_chart(fig)

    fig = go.Figure()
    for label, key in (("Reference", "obs"), ("Raw model", "raw"), ("TVC-corrected", "tvc")):
        fig.add_scatter(
            y=synth[key][:365], name=label, mode="lines", line=dict(width=1.1, color=SYNTH_COLORS[label])
        )
    fig.update_layout(
        title="First 365 validation days",
        xaxis_title="Day",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig)
