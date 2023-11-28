"""
Script to plot Figure 4-5

Author: Yawen Shao, created on May 30, 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 12,
        }

matplotlib.rc('font', **font)


def box_plot_prct_improve_3plots(data1, data2, data3, models, color1, color2, color3, mcolor, ylabel, title, stat, var):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9,8))
    letters=['a)','b)']
    
    for r in range(len(data1)):
        if r == 1:
            ax_tick = []
        
        for k in range(len(data1[r].mod)):
            # For data1
            b1=ax[r].boxplot(data1[r][var].isel(mod=k).values.flatten(),
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                positions=[0.12+k],
                widths=(0.13), 
                showmeans = True,
                showfliers = False,
                whiskerprops=dict(linewidth=2, color=color1[r]),
                capprops=dict(linewidth=0, color=color1[r]),
                boxprops=dict(linewidth=2, edgecolor=color1[r], facecolor=color1[r]), #, edgecolor=color
                medianprops=dict(linewidth=2, color=mcolor[r]),
                meanprops=dict(marker="D", markersize=0, markeredgecolor=mcolor[r], markerfacecolor=mcolor[r])
                )
            
            # For data2
            b2=ax[r].boxplot(data2[r][var].isel(mod=k).values.flatten(),
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                positions=[0.37+k],
                widths=(0.13), 
                showmeans = True,
                showfliers = False,
                whiskerprops=dict(linewidth=2, color=color2[r]),
                capprops=dict(linewidth=0, color=color2[r]),
                boxprops=dict(linewidth=2, edgecolor=color2[r], facecolor=color2[r]),
                medianprops=dict(linewidth=2, color=mcolor[r]),
                meanprops=dict(marker="D", markersize=0, markeredgecolor=mcolor[r], markerfacecolor=mcolor[r])
                )
    
            # For data3
            b3=ax[r].boxplot(data3[r][var].isel(mod=k).values.flatten(),
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                positions=[0.62+k],
                widths=(0.13), 
                showmeans = True,
                showfliers = False,
                whiskerprops=dict(linewidth=2, color=color3[r]),
                capprops=dict(linewidth=0, color=color3[r]),
                boxprops=dict(linewidth=2, edgecolor=color3[r], facecolor=color3[r]),
                medianprops=dict(linewidth=2, color=mcolor[r]),
                meanprops=dict(marker="D", markersize=0, markeredgecolor=mcolor[r], markerfacecolor=mcolor[r])
                )
    
            if r == 1:
                ax_tick.append(" ") 
                ax_tick.append(models[k])
                ax_tick.append(" ") 
    
        ax[r].axhline(y=0, color='grey', linestyle='dashed', linewidth=1)
        ax[r].legend([b1["boxes"][0], b2["boxes"][0], b3["boxes"][0]], ['Historical', 'ssp126', 'ssp585'], loc='lower right', fontsize=10)
        
        if stat[r] == 'WSDI6':
            ax[r].set_ylim([-90,100])
        else:
            ax[r].set_ylim([-35,100])
            
        ax[r].set_title(title[r], fontsize=16, weight='bold')
        ax[r].set_ylabel(ylabel, fontsize=14)
        ax[r].text(x=-0.3, y=105, s=letters[r], fontsize=15, fontweight='bold')
        
        if r == 1:
            ax[r].set_xticklabels(ax_tick, rotation=80, fontsize=10)
        else:
            ax[r].set_xticklabels([])
    
    fig.subplots_adjust(left=0.08,top=0.96,bottom=0.16,right=0.99,hspace=0.2)
    
    fig.savefig('./Figures/Figure_Prct_improve_'+stat[0]+'_'+stat[1]+'_MaT_notrend_9_all_model_'+var+'_global_0.1_scenarios.jpeg',dpi=300)
    
    
if __name__ == "__main__":
    models = pd.read_csv('./Data/model_name_ssp126585.csv', header=None)
    models = np.array(models.iloc[:,0])
    model_names = pd.read_csv('./Data/model_institution_ssp126585.csv', header=None)
    model_names = np.array(model_names.iloc[:,1])
    colors1 = [['salmon','lightgreen'],['palegreen','violet']]
    colors2 = [['lightcoral','limegreen'],['mediumseagreen','hotpink']]
    colors3 = [['indianred','forestgreen'], ['seagreen','deeppink']]
    colors_m = [['red','darkgreen'],['darkgreen','purple']]
    titles = [['Variance','lag-1 correlation'],['5-day lag correlation of 5-day average','Warm spell duration index']] #RMSD
    var_name = [['Var_30ymv','Lag1corr_30ymv'],['Lag5corr_30ymv','WSDI6']]
    
    for i, stat in enumerate(var_name):
        for var in ['tasmax']:
            data_hist = []
            data_ssp126 = []
            data_ssp585 = []
            for value in stat:
                if value == 'WSDI6':
                    data_hist.append(xr.open_dataset('./Data/Prct_improve_'+value+'_MaT_mean_corr_raw_new_clim_30y_2090_historical_notrend_9_all_model_'+var+'_global_0.1.nc'))
                    data_ssp126.append(xr.open_dataset('./Data/Prct_improve_'+value+'_MaT_mean_corr_raw_new_clim_30y_2090_ssp126_notrend_9_all_model_'+var+'_global_0.1.nc'))
                    data_ssp585.append(xr.open_dataset('./Data/Prct_improve_'+value+'_MaT_mean_corr_raw_new_clim_30y_2090_ssp585_notrend_9_all_model_'+var+'_global_0.1.nc'))        
                else:
                    data_hist.append(xr.open_dataset('./Data/Prct_improve_'+value+'_MaT_mean_corr_raw_historical_notrend_9_all_model_'+var+'_global_0.1.nc'))
                    data_ssp126.append(xr.open_dataset('./Data/Prct_improve_'+value+'_MaT_mean_corr_raw_ssp126_notrend_9_all_model_'+var+'_global_0.1.nc'))
                    data_ssp585.append(xr.open_dataset('./Data/Prct_improve_'+value+'_MaT_mean_corr_raw_ssp585_notrend_9_all_model_'+var+'_global_0.1.nc'))

            box_plot_prct_improve_3plots(data_hist, data_ssp126, data_ssp585, model_names, colors1[i], colors2[i], colors3[i], colors_m[i], 'Percentage improvement (%)', titles[i], stat, var)
            
            print(xr.concat(data_ssp585, dim='stat').median(dim=['lat','lon']).tasmax)
            
            