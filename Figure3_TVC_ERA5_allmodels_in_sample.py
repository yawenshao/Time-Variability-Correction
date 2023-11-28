"""
Script to conduct data analyssi and Figure 3 plotting

Author: Yawen Shao, created on May 30, 2023
"""

import numpy as np
import pandas as pd
from TVC_class import TVC
from statistics_tool import statistics
import matplotlib.pyplot as plt
import matplotlib

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 15,
        }

matplotlib.rc('font', **font)

def check_metrics(data, dates_TVC, yr_st, yr_end, base_st, base_end):
    stat_model = statistics(data)
    ### Variance
    variance = stat_model.sample_var_30y_mv()
    
    ### lag-1 correlation
    lag1 = stat_model.lag_corr_nonan_notr_30y_mv(1, 1)
    
    ### lag-5 correlation
    lag5 = stat_model.lag_corr_nonan_notr_30y_mv(5, 1)
    
    ### climate index WSDI
    mask_d = dates_TVC.where(dates_TVC.year >= base_st)
    mask_d = mask_d.where(mask_d.year <= base_end)
    base_data = data[~pd.isna(mask_d)]

    prcts = stat_model.base_period_percentile(base_data, base_st, base_end, 5)
    wsdi = stat_model.WSDI(6, False, prcts, yr_st, yr_end)
  
    return variance, lag1, lag5, np.nanmean(wsdi)


if __name__ == "__main__":
    models = pd.read_csv('./Data/model_name_ssp126585.csv', header=None)
    models = np.array(models.iloc[:,0])
    model_names = pd.read_csv('./Data/model_institution_ssp126585.csv', header=None)
    model_names = np.array(model_names.iloc[:,1])
    scale = [365,183,92,46,23,12,6,3,2]
    init = sum(scale)-len(scale)
    
    base_st, base_end, yr_st, yr_end = 1961, 1990, 1952, 2014
    dates = pd.date_range('1950-01-01', '2014-12-31')
    dates_TVC = dates[init:]
    
    K = len(models) # number of models
    
    # Initialise metric result array
    stat_TVC = np.full((K, 4), np.nan) # variance, rmse, lag1, lag5, CI
    stat_raw = np.full((K, 4), np.nan)
    stat_obs = np.full((4,), np.nan)
    
    obs = pd.read_csv('./Data/ERA5_obs_-37.5_144.5_tasmax_hist.csv', header=None)
    obs = np.array(obs).flatten()
    
    ### Metrics for obs
    stat_obs[0], stat_obs[1], stat_obs[2], stat_obs[3] = check_metrics(obs[init:], dates_TVC, yr_st, yr_end, base_st, base_end)
    
    raw_all = pd.read_csv('./Data/ERA5_allmodels_-37.5_144.5_tasmax_hist.csv', header=None)
    raw_all = np.array(raw_all)
    
    # mean correct raw predictions
    raw_all = raw_all + np.nanmean(obs) - np.nanmean(raw_all, axis=0)
 

    ### Post-process models individually using TVC       
    Z_hist_TVC_all = []
    for m in range(K):
        TVC_model = TVC(obs, raw_all[:,m], raw_all[:,m], scale)
        Z_hist = TVC_model.TVC_postprocess()
        Z_hist_TVC = np.sum(Z_hist, axis=1)        
        Z_hist_TVC_all.append(Z_hist_TVC)
        
        stat_raw[m, 0], stat_raw[m, 1], stat_raw[m, 2], stat_raw[m, 3] = check_metrics(raw_all[init:,m], dates_TVC, yr_st, yr_end, base_st, base_end)
        stat_TVC[m, 0], stat_TVC[m, 1], stat_TVC[m, 2], stat_TVC[m, 3] = check_metrics(Z_hist_TVC, dates_TVC, yr_st, yr_end, base_st, base_end)
    
    Z_hist_TVC_all = np.array(Z_hist_TVC_all)
    np.savetxt('./Data/TVC_ERA5_allmodels_-37.5_144.5_tasmax_hist.csv', Z_hist_TVC_all.T, fmt = '%5.5f', delimiter=",")
    
    
    ############# Figure 3
    #### Plot boxplot-like scatter plots of statistics
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11,9))
    cols = ['orange','limegreen','dodgerblue','lightcoral','hotpink']
    syms = ['o','^','s','p','P']
    lims = [[[22,60],[5.3,7.3],[0.73,0.95],[0.73,0.915],[0.5,6.1]],[[12,22],[3.6,4.65],[0.725,0.875],[0.7,0.85],[0,2.2]]]
    ylabels = ["Variance ($^\circ$C $^2$)", 'lag-1 correlation', '5-day lag correlation', 'Warm spell duration index']
    letters = ['a)','b)','c)','d)']
    l_coords = [104.5,0.914,0.93,9.15] 
    legends = []
    X = np.linspace(1,14,14,dtype=int)

    for c in range(stat_TVC.shape[1]):
        for m in range(K):
            # For raw and TVC
            if c == 0:
                b1 = ax[c//2,c%2].scatter(X[m%5], stat_raw[m,c], s=70, c=cols[m//5], marker=syms[m%5], label=model_names[m])
                ax[c//2,c%2].scatter(X[m%5+8], stat_TVC[m,c], s=70, c=cols[m//5], marker=syms[m%5], label=model_names[m])
                legends.append(b1)
            else:
                ax[c//2,c%2].scatter(X[m%5], stat_raw[m,c], s=70, c=cols[m//5], marker=syms[m%5])
                ax[c//2,c%2].scatter(X[m%5+8], stat_TVC[m,c], s=70, c=cols[m//5], marker=syms[m%5])
    

        ax[c//2,c%2].axhline(stat_obs[c], color='black', alpha=0.7, linestyle='dashed', linewidth=1)
        ax[c//2,c%2].set_ylabel(ylabels[c], fontsize=17)
        ax[c//2,c%2].set_xticks([3,11])
        
        if c//2 == 1:
            ax[c//2,c%2].set_xticklabels(['Mean-corrected Raw','TVC'], fontsize=17)
        else:
            ax[c//2,c%2].set_xticklabels([])
        
        ax[c//2,c%2].set_xlim([0,14])
        ax[c//2,c%2].text(x=0.2, y=l_coords[c], s=letters[c], fontsize=17, fontweight='bold')
        
    
    legend = plt.legend(handles=legends, bbox_to_anchor=(1, -0.14),
              ncol=6, fontsize=11, handletextpad = 0.4,labelspacing=0.35,handlelength=1.3)
    legend._legend_box.sep = 8
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor('k')

    
    fig.subplots_adjust(left=0.09,top=0.97,bottom=0.16,right=0.99,hspace=0.18,wspace=0.25)
    fig.savefig('./Figures/Figure3_Boxlike_scatters_-37.5_144.5_historical_metrics_mean_corrected.jpeg',dpi=300)
    