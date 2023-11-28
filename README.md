# Time-Variability-Correction

This repository includes the Python scripts and relevant data used to build the Time Variability Correction model, conduct the analysis, and create figures for the manuscript: Yawen Shao, Craig H. Bishop, Sanaa Hobeichi, Nidhi Nishant, Gab Abramowitz, Steven Sherwood. Time Variability Correction of CMIP6 Climate Change Projections. Accepted in Journal of Advances in Modelling Earth Systems. 

Cite DOI: 10.5281/zenodo.10212122

## Script Contents Overview:
### Time Variability Correction model and verification metrics
‘TVC_class.py’: Python class implementing the Time Variability Correction (TVC) method. Detailed function comments are included in the script.

‘statistics_tool.py’: Python class to compute verification metrics for observed, raw and TVC post-processed data. Metrics include sample variance and lag correlations relative to the 30-year running mean, as well as the warm spell duration index. The calculation of the warm spell duration index follows the general procedures used by the Expert Team on Climate Change Detection and Indices (ETCCDI).

‘TVC_case_study.py’: contains a test case demonstrating the usage of ‘TVC_class.py’ and ‘statistics_tool.py’ for post-processing data and compute statistics.

### Figure plotting
‘Figure1-2_S6-7_TVC_ACCESS-ESM1-5_ERA5_case_in_sample.py’: codes applying TVC to correct ACCESS-ESM1-5 time series for a test case, plotting covariance matrices (Figure 1) and time series of the 92-day time scale and combined series for observed data and mean-corrected raw model (Figure 2).

‘Figure3_TVC_ERA5_allmodels_in_sample.py’: codes correcting all CMIP6 models for a test case, computing verification metrics, and plotting the scatter plots of metric results (Figure 3).

‘Figure4-5_box_plot_prct_improve_all_scenarios_2plots.py’: illustrates the box plot showing the percentage improvements in mean absolute error of all verification metrics (Figure 4 and 5).

‘Figure6_global_aggregate_scatter_plots.py’: calculates and plots globally weighted absolute difference between TVC and mean-corrected raw variance at each time scale (Figure 6).

‘Figure7_one_model_lat_aggregate_heatmap.py’: computes and plots the absolute difference between ERA5 reanalysis and mean-corrected raw model in time-scale-dependent variance averaged at each latitude. Data used to plot Figure 7 are on Zenodo (10.5281/zenodo.10212122).

‘Figure8_S8_global_map_plot_6plots.py’: plots the difference in variance and lag correlations between TVC and mean-corrected raw predictions across the globe.

‘FigureS1-4_example_covariance_lag_correlation.py’: illustrates a ‘toy’ example to demonstrate how improving temporal variability at different time scales could enhance temporal correlations. Includes codes for for plotting covariance matrix heatmaps (Figure S1 and S3), eigenvectors of the covariance (Figure S2) and eigenvalue spectrums (Figure S4).

‘AR1_standard_Gaussian_nocycle_30_500years_check_statistics.py’: codes for modelling Gaussian AR1 process for 30-year and 500-year periods, improving lag correlations and computing the verification metrics.

‘FigureS5_Plot_30_500years_statistics.py’: plots metric results from ‘AR1_standard_Gaussian_nocycle_30_500years_check_statistics.py’ and computes the relative error and percentage improvement for the metric results.


The following Python modules are needed to run the above scripts:
- numpy
- pandas
- xarray
- scipy
- calendar
- operator
- datetime
- math
- matplotlib
- mpl_toolkits
- seaborn

