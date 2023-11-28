"""
Script to plot Figure S5

Author: Yawen Shao, created on May 30, 2023
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 12,
        }

matplotlib.rc('font', **font)

if __name__ == '__main__':
    #### Box plot showing the spread of the 100 realisations
    ### 2 rows * 3 columns
    stats = ['lag1Corr','lag5Corr','wsdi']
    varr = ['Raw','Corrected']
    yrs = ['30','500']

    labels = ['lag-1 correlation','5-day lag correlation','Warm spell duration index']
    
    colors = [['limegreen','forestgreen'],['mediumseagreen','seagreen'],['hotpink','deeppink']]
    colors_m = ['forestgreen','darkgreen','purple']
    
    letters = ['a)','b)','c)','d)','e)','f)']
    ylims = [[0.78, 0.92],[0.44,0.75],[4,19]] 
    text_y = [0.923, 0.757, 19.42]
    legends = []
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,8))
    for r, y in enumerate(yrs):
        for c, stat in enumerate(stats):
            obs = pd.read_csv('./Data/obs_'+y+'y_0.8_0.9_'+stat+'_AR1_100.csv', header=None)
            obs = np.array(obs).flatten()
            
            for i, mod in enumerate(varr):
                data = pd.read_csv('./Data/'+mod+'_new_'+y+'y_0.8_0.9_'+stat+'_AR1_100.csv', header=None)
                data = np.array(data).flatten()
                
                if i == 0:
                    raw_data = data.copy()
                elif i == 1:
                    corr_data = data.copy()
                
                a1 = ax[r,c].boxplot(data,
                    vert=True,
                    patch_artist=True,
                    positions=[0.05+0.7*i],
                    widths=(0.2), 
                    showmeans = True,
                    showfliers = False,
                    whiskerprops=dict(linewidth=2, color=colors[c][i]),
                    capprops=dict(linewidth=2, color=colors[c][i]),
                    boxprops=dict(linewidth=2, edgecolor=colors[c][i], facecolor='white'),
                    medianprops=dict(linewidth=2, color=colors_m[c]),
                    meanprops=dict(marker="D", markersize=0, markeredgecolor=colors_m[c], markerfacecolor=colors_m[c])
                    )          
            
            ax[r,c].axhline(y=np.mean(obs), color='grey', linestyle='dashed', linewidth=1)
            ax[r,c].fill_between([-0.35,1.35],np.mean(obs)-np.std(obs),np.mean(obs)+np.std(obs), color='grey', alpha=0.3)
            
            if c == 1:
                ax[r,c].set_title(yrs[r]+'-years', fontsize=16, weight='bold', pad=10)
                
            ax[r,c].set_ylabel(labels[c], fontsize=14)
            ax[r,c].text(x=-0.35, y=text_y[c], s=letters[len(stats)*r+c], fontsize=15, fontweight='bold')
            ax[r,c].set_ylim(ylims[c])
            ax[r,c].set_xlim([-0.35,1.35])
            
            if r == 1:
                ax[r,c].set_xticklabels(varr, fontsize=14)
            else:
                ax[r,c].set_xticklabels([])
    
            # calculate the errors
            if c == 1: # for 30-year
                # Error relative to the true value
                print('----------')
                
                lag5_obs = np.mean(obs)
                lag5_rmse = np.sqrt(np.sum(np.power(data-lag5_obs, 2))/len(data)) ##obs
                error_prct = lag5_rmse/lag5_obs*100
                print('lag5: '+str(error_prct))
                
                # percentage improvement
                raw_MAE = np.mean(np.abs(raw_data-obs))
                corr_MAE = np.mean(np.abs(corr_data-obs))
                prct_imprv = (raw_MAE-corr_MAE)/raw_MAE * 100
                print('prct improve: '+str(prct_imprv))
    
            if c == 2: # for longer period
                # Error relative to the true value
                wsdi_obs = np.mean(obs)
                corr_rmse = np.sqrt(np.sum(np.power(data-wsdi_obs, 2))/len(data))  ## obs
                error_prct = corr_rmse/wsdi_obs*100
                print('wsdi: '+str(error_prct))
    
                # percentage improvement
                raw_MAE = np.mean(np.abs(raw_data-obs))
                corr_MAE = np.mean(np.abs(corr_data-obs))
                prct_imprv = (raw_MAE-corr_MAE)/raw_MAE * 100
                print('prct improve: '+str(prct_imprv))
    
                
            fig.subplots_adjust(left=0.05,top=0.96,bottom=0.05,right=0.99,hspace=0.25,wspace=0.2)
            fig.savefig('./Figures/FigureS5_AR1_500_30years_boxplot_uncertainty.jpeg',dpi=300)
        