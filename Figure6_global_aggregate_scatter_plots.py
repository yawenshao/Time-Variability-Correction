"""
Script to plot Figure 6

Author: Yawen Shao, created on May 30, 2023
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import cos

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,9))
cols = ['orange','limegreen','dodgerblue','lightcoral','hotpink']
syms = ['o','^','s','p','P']
lims = [[[22,60],[5.3,7.3],[0.73,0.95],[0.73,0.915],[0.5,6.1]],[[12,22],[3.6,4.65],[0.725,0.875],[0.7,0.85],[0,2.2]]]
letters = ['a)','b)']
l_coords = [104.5,0.914,0.93,9.15] 
legends = []
X = np.linspace(1,76,76,dtype=int)
model_names = pd.read_csv('./Data/model_institution_ssp126585.csv', header=None)
model_names = np.array(model_names.iloc[:,1])
xlabels = ['365-day', '183-day','92-day','46-day','23-day','12-day','6-day','3-day', '2-day', '1-day', 'Combined']

for r, ssp in enumerate(['historical','ssp585']):  #,'ssp126','ssp585'
    raw = xr.open_dataset('./Data/Var_all_global_'+ssp+'_tasmax_raw.nc')
    TVC = xr.open_dataset('./Data/Var_all_global_'+ssp+'_tasmax_TVC.nc')
    
    diff = (TVC - raw)['tasmax']
    
    diff_new = []
    cos_sum = 0
    for l in diff.lat.values: # calculate absolute average
        data = abs(diff.sel(lat=l)*cos(abs(l)*np.pi/180)).sum(dim='lon')
        cos_sum += 360*cos(abs(l)*np.pi/180)
        diff_new.append(data)
    
    diff_new = xr.concat(diff_new, dim='lat')
    diff_sum = diff_new.sum(dim='lat')/cos_sum
    
    for ss in range(len(diff_sum.s)):
        for m in range(len(diff_sum.model)):
            # For raw and TVC
            if r == 0 and ss == 0:
                b1 = ax[r].scatter(X[m%5+7*ss], diff_sum.isel(s=ss,model=m), s=55, c=cols[m//5], marker=syms[m%5], label=model_names[m])
                legends.append(b1)
            else:
                ax[r].scatter(X[m%5+7*ss], diff_sum.isel(s=ss,model=m), s=55, c=cols[m//5], marker=syms[m%5])

    ax[r].set_xlim([0,76])
    ax[r].set_ylim([-0.5,18])

    if r == 1:
        ax[r].set_xticklabels(xlabels, fontsize=17)
    else:
        ax[r].set_xticklabels([])
            
    
    ax[r].set_xticks(np.linspace(3, 73, 11,dtype=int))
    ax[r].set_title(ssp, fontsize=19, weight='bold')
    ax[r].text(x=-0.2, y=18.5, s=letters[r], fontsize=18, fontweight='bold')
    
legend = plt.legend(handles=legends, bbox_to_anchor=(1.02, -0.13),
          ncol=6, fontsize=11, handletextpad = 0.4,labelspacing=0.35,handlelength=1.3)
legend._legend_box.sep = 8
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('k')
ax[1].text(x=-5.5, y=4, s='Absolute Difference (TVC-Raw) in Variance ($^\circ$C $^2$)', rotation=90, fontsize=17)

fig.subplots_adjust(left=0.08,top=0.96,bottom=0.16,right=0.975,hspace=0.2)
fig.savefig('./Figures/Figure6_global_aggregate_scatters_variance.jpeg',dpi=300)