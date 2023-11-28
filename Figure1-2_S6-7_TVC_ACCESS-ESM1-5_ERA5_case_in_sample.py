"""
Script to apply time variability correction method to correct ACCESS-ESM1-5 time series
for a test case, as well as plot the covariance matrices and time series of the 92-day
time scale and combined series for obs and raw model

Author: Yawen Shao, created on May 30, 2023
"""

import numpy as np
import pandas as pd
from TVC_class import TVC
import matplotlib.pyplot as plt
import matplotlib


font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 15,
        }

matplotlib.rc('font', **font)


if __name__ == "__main__":    
    colors_raw = ['lightcoral', 'limegreen', 'mediumseagreen', 'violet']
    colors_TVC = ['indianred', 'forestgreen', 'seagreen', 'hotpink']
    ylabels = ["Variance ($^\circ$C $^2$)", 'lag-1 correlation', '5-day lag correlation', 'Warm spell duration index'] #'5-day lag correlation of 5-day average'
    
    scale = [365,183,92,46,23,12,6,3,2]
    init = sum(scale)-len(scale)
    dates = pd.date_range('1950-01-01', '2014-12-31')
    
    # Import data
    obs = pd.read_csv('./Data/ERA5_obs_-37.5_144.5_tasmax_hist.csv', header=None)
    obs = np.array(obs).flatten()
    
    raw_all = pd.read_csv('./Data/ERA5_ACCESS-ESM1-5_-37.5_144.5_tasmax_hist.csv', header=None)
    raw_all = np.array(raw_all).flatten()
    
    # Apply TVC method
    TVC_model = TVC(obs, raw_all, raw_all, scale)
    Z_hist = TVC_model.TVC_postprocess()
    
    # Get decomposed obs and mean-corrected raw model series
    Y = TVC_model.get_decomposition('obs')
    X = TVC_model.get_decomposition('train')
    X_corr = X + np.mean(Y, axis=0) - np.mean(X, axis=0)    
 
    
    ########## Figure 1
    ### Plot covariance heatmap
    obs_cov = TVC_model.get_TVC_covariance('obs')
    raw_cov = TVC_model.get_TVC_covariance('train')
    
    heat_title = ['Observations','Mean-corrected Raw Predictions']
    cov_all = [obs_cov, raw_cov]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5.8))
    label = ['365-day', '183-day','92-day','46-day','23-day','12-day','6-day','3-day', '2-day', '1-day']
    letters = ['a)','b)']
    
    nn = 16
    cmap_get = plt.get_cmap('RdBu_r',nn)
    cmap = matplotlib.colors.ListedColormap([cmap_get(i) for i in range(3,8)] + [cmap_get(i) for i in range(8,nn-1)])
    cmap.set_over(cmap_get(nn-1))
    cmap.set_under(cmap_get(2))
   
    levels = np.array([-10,-5,-1,-0.5,-0.1,0,0.1,0.5,1,5,10,20,25])
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
   
    for c in range(2):
        im = ax[c].imshow(cov_all[c], cmap=cmap, norm=norm)
       
        # Show all ticks and label them with the respective list entries
        ax[c].set_xticks(np.arange(10))
        ax[c].set_yticks(np.arange(10))
       
        ax[c].set_xticklabels(label, rotation=45, fontsize=12)
       
        if c == 0:
            ax[c].set_yticklabels(label, fontsize=12)
        else:
            ax[c].set_yticklabels([])
       
        # Loop over data dimensions and create text annotations.
        for s in range(len(label)):
            for j in range(len(label)):
                if s == 2 and j ==2:
                    cc = 'white'
                elif s == 3 and j ==3:
                    cc = 'white'
                else:
                    cc = 'black'
                text = ax[c].text(j, s, np.round(cov_all[c][s, j],2),
                              ha="center", va="center", color=cc, size=11)
   
        ax[c].set_title(heat_title[c], fontsize=15, weight='bold')
        ax[c].text(x=-0.4, y=-0.74, s=letters[c], fontsize=14, fontweight='bold')
       
    cbaxes = fig.add_axes([0.18, 0.06, 0.7, 0.03]) #left, bottom, width, height
    cbar = fig.colorbar(im, orientation='horizontal', cax=cbaxes,
                        extend='max',
                        shrink=0.6,
                        ticks=levels
                        )
    cbar.ax.xaxis.set_tick_params(pad=5)
    cbar.ax.tick_params(labelsize=12)
   
    fig.subplots_adjust(left=0.075,top=0.96,bottom=0.2,right=0.98, wspace=0.1)
    fig.savefig('./Figures/Figure1_ERA5_-37.5_144.5_ACCESS-ESM1-5_obs_raw_cov_heatmap.jpeg', dpi=300)
 

    ########## Figure 2
    ### Plot time series of 92-day time scale and combined case
    # Plot 1965-1974 for 92-day time scale and plot Year 1965 for combined time series
    end_ind = [3652+5479-init, 365+5479-init]
    start_dates = dates[5479]
    end_dates = [dates[3652+5479-1], dates[365+5479-1]]
    start_ind = [5479-init, 5479-init]
    titles = ['92-day', 'Combined']
    letters = ['a)','b)']
    y_coords = [13,39.2]
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,8.5))
        
    for i in range(2):
        years = pd.date_range(start=start_dates, end=end_dates[i], freq='D')
        
        if titles[i] == 'Combined':
            ax[i].plot(years, np.sum(Y[start_ind[i]:end_ind[i],:], axis=1), linewidth=0.9, color='red', label='ERA5',zorder=1)
            ax[i].plot(years, np.sum(X_corr[start_ind[i]:end_ind[i],:], axis=1), linewidth=0.9, color='grey', label='MC-raw',zorder=3)
            ax[i].plot(years, np.sum(Z_hist[start_ind[i]:end_ind[i],:], axis=1), linewidth=0.9, color='blue', label='TVC-corrected',zorder=2)
        else:
            ax[i].plot(years, Y[start_ind[i]:end_ind[i],i+2], linewidth=0.9, color='red', label='ERA5',zorder=1)
            ax[i].plot(years, X_corr[start_ind[i]:end_ind[i],i+2], linewidth=0.9, color='grey', label='MC-raw',zorder=3)
            ax[i].plot(years, Z_hist[start_ind[i]:end_ind[i],i+2], linewidth=0.9, color='blue', label='TVC-corrected',zorder=2)   
            
        ax[i].set_xlim([years[0],years[-1]])         
        ax[i].set_ylabel('Temperature ($^\circ$C)', fontsize=16)
        ax[i].set_title(titles[i], fontsize=18, fontweight='bold')
        ax[i].text(x=years[0], y=y_coords[i], s=letters[i], fontsize=18, fontweight='bold')
        
        if i == 1:
            ax[i].set_xticks(['1965-01','1965-03','1965-05','1965-07','1965-09','1965-11','1966-01'])
            ax[i].set_xticklabels(['1965-01','1965-03','1965-05','1965-07','1965-09','1965-11','1966-01'])
    
    ax[i].set_xlabel('Date', fontsize=16)

    fig.subplots_adjust(left=0.07,top=0.96,bottom=0.07,right=0.96, hspace=0.28)
    fig.savefig('./Figures/Figure2_ERA5_-37.5_144.5_ACCESS-ESM1-5_obs_raw_TVC_time_series.jpeg',dpi=300)
    

    ########## Figure S6-7
    #### Plot time series of all time scales (split into 2 plots)
    # all years, 25yr, 25yr, 5yr, 5yr, 2yr, 2yr, 2yr, 2yr, 2yr
    end_ind = [len(obs[init:]), 9131, 9131, 1826, 1826, 730, 730, 730, 730, 730, 730]
    start_dates = dates[init]
    titles = ['365-day', '183-day','92-day','46-day','23-day','12-day','6-day','3-day', '2-day', '1-day', 'Combined']

    
    p = 0
    rr = [6,5]
    while p < 2:
        if p == 0:
            fig, ax = plt.subplots(nrows=rr[p], ncols=1, figsize=(13,19))
            pp = 0
        elif p == 1:
            fig, ax = plt.subplots(nrows=rr[p], ncols=1, figsize=(13,16))
            pp = rr[p-1]
        
        for i in range(rr[p]):
            years = pd.date_range(start=start_dates, end=start_dates+timedelta(days=end_ind[i+pp]-1), freq='D')
            
            if titles[i+pp] == 'Combined':
                ax[i].plot(years, np.sum(Y[:end_ind[i+pp],:], axis=1), linewidth=0.9, color='red', label='Obs',zorder=1)
                ax[i].plot(years, np.sum(X_corr[:end_ind[i+pp],:], axis=1), linewidth=0.9, color='grey', label='MC-raw',zorder=3)
                ax[i].plot(years, np.sum(Z_hist[:end_ind[i+pp],:], axis=1), linewidth=0.9, color='blue', label='TVC',zorder=2)
            else:
                ax[i].plot(years, Y[:end_ind[i+pp],i+pp], linewidth=0.9, color='red', label='Obs',zorder=1)
                ax[i].plot(years, X_corr[:end_ind[i+pp],i+pp], linewidth=0.9, color='grey', label='MC-raw',zorder=3)
                ax[i].plot(years, Z_hist[:end_ind[i+pp],i+pp], linewidth=0.9, color='blue', label='TVC',zorder=2)   
                
            ax[i].set_xlim([years[0],years[-1]])         
            ax[i].set_ylabel('Temperature ($^\circ$C)', fontsize=16)
            ax[i].set_title(titles[i+pp], fontsize=18)
            
        ax[0].legend(fontsize=14)
        
        p += 1
        ax[i].set_xlabel('Date', fontsize=16)
    
        fig.subplots_adjust(left=0.06,top=0.98,bottom=0.035,right=0.99, hspace=0.35)
        fig.savefig('./Figures/FigureS'+str(5+p)+'_ERA5_-37.5_144.5_ACCESS-ESM1-5_obs_raw_TVC_time_series_plot.jpeg',dpi=300)