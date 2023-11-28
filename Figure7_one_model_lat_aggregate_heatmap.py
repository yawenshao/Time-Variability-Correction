"""
Script to plot Figure 7

Author: Yawen Shao, created on May 30, 2023
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16,
        }
matplotlib.rc('font', **font)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,10))
xlabels = ['365-day', '183-day','92-day','46-day','23-day','12-day','6-day','3-day', '2-day', '1-day', 'Combined']
model = 'ACCESS-ESM1-5'
heat_title = [model,'Averaged across all CMIP6 models']
label = 'Absolute Difference (Obs-Raw) in Variance ($^\circ$C $^2$)'

nn = 11
cmap_get = plt.get_cmap('Reds',nn)
cmap = matplotlib.colors.ListedColormap([cmap_get(i) for i in range(1,nn-1)])
cmap.set_over(cmap_get(nn-1))
cmap.set_under(cmap_get(0))

levels = np.array([0,0.5,1,2,5,10,20,30,40,50])
norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
letters = ['a)','b)']


for ssp in ['historical']:  #,'ssp126','ssp585'
    raw = xr.open_dataset('./Data/Var_all_global_'+ssp+'_tasmax_raw.nc')
    TVC = xr.open_dataset('./Data/Var_all_global_'+ssp+'_tasmax_TVC.nc')
    
    diff = (TVC - raw)['tasmax']
    
    diff_m = abs(diff).mean(dim=['lon']).sel(model=model).values[::-1,:]
    diff_all = abs(diff).mean(dim=['lon','model']).values[::-1,:]
    
    lats = diff.lat.values[::-1]
    
    cbaxes = [0.18, 0.06, 0.7, 0.03] #left, bottom, width, height
    cax = inset_axes(ax[0],
                 width="140%",
                 height="4%", 
                 loc='lower left',
                 bbox_to_anchor=(0.325, -0.18, 1, 1),
                 bbox_transform=ax[0].transAxes,
                 borderpad=0
                 )

    sns.heatmap(diff_m, ax=ax[0], cbar_ax=cax, cmap=cmap, norm=norm, cbar_kws={'orientation': 'horizontal','ticks':levels, 'extend':'max', 'label':label})
    sns.heatmap(diff_all, ax=ax[1], cbar=False, cmap=cmap, norm=norm)

    # Show all ticks and label them with the respective list entries
    for i in range(2):
        ax[i].set_xticks(np.arange(11))
        ax[i].set_yticks(np.arange(4,179,10))
    
        ax[i].set_xticklabels(xlabels, rotation=45, fontsize=16)
    
        if i == 0:
            ax[i].set_yticklabels(lats[4::10], fontsize=16)
        else:
            ax[i].set_yticklabels([])
    
        ax[i].set_title(heat_title[i], fontsize=22, weight='bold')
        ax[i].text(x=-0.6, y=-2.2, s=letters[i], fontsize=20, fontweight='bold')

    ax[0].set_ylabel('Latitude', fontsize=20)

fig.subplots_adjust(left=0.08,top=0.96,bottom=0.2,right=0.99, wspace=0.1)
fig.savefig('./Figures/Figure7_'+model+'_global_aggregate_latitude_variance_heatmap.jpeg', dpi=300)
