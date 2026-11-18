# Paper reproduction scripts

Scripts used for the figures in Shao et al. (2024), https://doi.org/10.1029/2023MS003640.
Their analysis and plotting code is unchanged from the published version; only imports, the font
setup and file paths now go through the `tvc` package. Inputs are read from `Data/` and figures are
written to `Figures/` (created automatically). Set `TVC_DATA_DIR` / `TVC_FIGURES_DIR` to use other locations.

Install the research extras first: `python -m pip install -e ".[research]"`, then run from any directory,
for example `python scripts/paper/fig03_all_models_site_metrics.py`.

| Script | Output | Previous name |
|---|---|---|
| `tvc_case_study.py` | Metrics for the ACCESS-ESM1-5 site case (no figure) | `TVC_case_study.py` |
| `fig01_02_S6_S7_case_study_access_esm1_5.py` | Figures 1, 2, S6, S7 | `Figure1-2_S6-7_TVC_ACCESS-ESM1-5_ERA5_case_in_sample.py` |
| `fig03_all_models_site_metrics.py` | Figure 3; writes `Data/TVC_ERA5_allmodels_-37.5_144.5_tasmax_hist.csv` | `Figure3_TVC_ERA5_allmodels_in_sample.py` |
| `fig04_05_prct_improve_boxplots.py` | Figures 4, 5 | `Figure4-5_box_plot_prct_improve_all_scenarios_2plots.py` |
| `fig06_global_aggregate_scatter.py` | Figure 6 | `Figure6_global_aggregate_scatter_plots.py` |
| `fig07_lat_aggregate_heatmap.py` | Figure 7 | `Figure7_one_model_lat_aggregate_heatmap.py` |
| `fig08_S8_global_maps.py` | Figures 8, S8 | `Figure8_S8_global_map_plot_6plots.py` |
| `figS1_S4_toy_covariance_example.py` | Figures S1–S4 | `FigureS1-4_example_covariance_lag_correlation.py` |
| `ar1_30_500y_statistics.py` | AR(1) experiment CSVs in `Data/` | `AR1_standard_Gaussian_nocycle_30_500years_check_statistics.py` |
| `figS5_ar1_statistics_boxplots.py` | Figure S5 | `FigureS5_Plot_30_500years_statistics.py` |

## Inputs not shipped in `Data/`

- Figures 6 and 7: `Var_all_global_{historical,ssp585}_tasmax_{raw,TVC}.nc`.
- Figures 8 and S8: `World_Continents.shp` (and its sidecar files), from
  https://hub.arcgis.com/datasets/CESJ::world-continents/explore.
