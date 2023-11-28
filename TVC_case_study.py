"""
Script to demonstrate how to use the Time Variability Correction class and do data analysis

Author: Yawen Shao, created on May 30, 2023
"""

import numpy as np
import pandas as pd
from TVC_class import TVC
from statistics_tool import statistics


if __name__ == "__main__":
    scale = [365,183,92,46,23,12,6,3,2]
    
    # Import data
    obs = pd.read_csv('ERA5_obs_-37.5_144.5_tasmax_hist.csv', header=None)
    obs = np.array(obs).flatten()
    
    raw_all = pd.read_csv('ERA5_ACCESS-ESM1-5_-37.5_144.5_tasmax_hist.csv', header=None)
    raw_all = np.array(raw_all).flatten()
    
    # Apply TVC method
    TVC_model = TVC(obs, raw_all, raw_all, scale)
    Z_hist = TVC_model.TVC_postprocess()
    Z_hist_TVC = np.sum(Z_hist, axis=1)
    
    #############################
    # Data analysis for TVC post-processed data
    stat_model = statistics(Z_hist_TVC)
    
    ### Variance
    variance = stat_model.sample_var_30y_mv()
    
    ### lag-1 correlation
    lag1 = stat_model.lag_corr_nonan_notr_30y_mv(1, 1)
    
    ### lag-5 correlation
    lag5 = stat_model.lag_corr_nonan_notr_30y_mv(5, 1)
    
    ### climate index WSDI
    # Base period: 1961-1990
    dates = pd.date_range('1950-01-01', '2014-12-31')
    init = sum(scale)-len(scale)
    dates_TVC = dates[init:]
    base_st = 1961
    base_end = 1990
    yr_st = 1952
    yr_end = 2014
    
    mask_d = dates_TVC.where(dates_TVC.year >= base_st)
    mask_d = mask_d.where(mask_d.year <= base_end)
    base_data = Z_hist_TVC[~pd.isna(mask_d)]

    prcts = stat_model.base_period_percentile(base_data, base_st, base_end, 5)
    wsdi = stat_model.WSDI(6, False, prcts, yr_st, yr_end)
    wsdi_avg = np.mean(wsdi)
