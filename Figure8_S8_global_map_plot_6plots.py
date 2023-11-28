"""
Script to plot Figure 8 and Figure S8
The shapefile of the global map ('./Data/World_Continents.shp') is downloadable at https://hub.arcgis.com/datasets/CESJ::world-continents/explore

Author: Yawen Shao, created on May 30, 2023
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

shpFilePath = './Data/World_Continents.shp' # Please self-download the file from https://hub.arcgis.com/datasets/CESJ::world-continents/explore
shape_feature = ShapelyFeature(Reader(shpFilePath).geometries(),ccrs.Mercator(),edgecolor='black',facecolor='none')

font = {
        'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16,
        }
matplotlib.rc('font', **font)
rcParams['axes.linewidth'] = 1


lon = np.linspace(-180,180,num=361,dtype=float)
lon = np.round(lon,1)
lat = np.linspace(-90,90,num=181,dtype=float)
lat = np.round(lat,1)
import matplotlib
print(matplotlib.get_cachedir())

color = ['RdBu_r','BrBG','PiYG']


bar_label = ['Difference (TVC-Raw) ($^\circ$C $^2$)','Difference (TVC-Raw)','Difference (TVC-Raw)']
titles = ['Variance','Lag-1 correlation','Lag-5 correlation']
bar_y = [0.72,0.41,0.1]
letters = ['a)','b)','c)','d)','e)','f)']

for c, var in enumerate(['all_model', 'ACCESS-ESM1-5']):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(11,13), subplot_kw={'projection': ccrs.Mercator()})  
    
    if c == 0:
        levels = levels = [[-100,-50,-30,-10,-5,-2,-1,0,1,2,5,10,30,50,100],[-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2],[-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2]]
        num = [16, 10, 10]
    elif c == 1:
        levels = [[-150,-100,-50,-30,-10,-5,-2,-1,0,1,2,5,10,30,50,100,150],[-0.3,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.3],[-0.3,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.3]]
        num = [18, 12, 12]

    for i, name in enumerate(['Var_30ymv','Lag1corr_30ymv','Lag5corr_30ymv']):
        cmap_get = plt.get_cmap(color[i], num[i])
        cmap = matplotlib.colors.ListedColormap([cmap_get(k) for k in range(1,num[i]-1)])
        cmap.set_over(cmap_get(num[i]-1))
        cmap.set_under(cmap_get(0))
        norm = matplotlib.colors.BoundaryNorm(levels[i], cmap.N)
        
        for r, ssp in enumerate(['historical','ssp585']):
            if c == 0:
                TVC = xr.open_dataset('./Data/ERA5_avg_'+name+'_MaT_'+ssp+'_9_'+var+'_tasmax_global_test.nc')
                raw = xr.open_dataset('./Data/ERA5_avg_'+name+'_MaT_'+ssp+'_mean_corr_raw_9_'+var+'_tasmax_global_test.nc')
            elif c == 1:
                TVC = xr.open_dataset('./Data/'+name+'_ERA5_'+ssp+'_notrend_9_'+var+'_tasmax_global_test.nc')
                raw = xr.open_dataset('./Data/'+name+'_ERA5_'+ssp+'_notrend_9_'+var+'_mean_corr_raw_tasmax_test.nc')
            
            diff = TVC - raw
            
            # convert [0,360] longitude to [-180, 180]
            diff = diff.assign_coords(lon=(((diff.lon + 180) % 360) - 180)).sortby('lon')

            im = ax[i,r].pcolormesh(
                lon,
                lat,
                diff['tasmax'],
                norm=norm,
                cmap=cmap
            )

            ax[i,r].add_feature(shape_feature, linewidth=0.5)
            ax[i,r].set_extent([-180,180,-90,90],ccrs.Mercator())
            ax[i,r].set_title(ssp+'  '+titles[i], fontsize=18, weight='bold', pad=10)
            ax[i,r].text(x=-179, y=102, s=letters[2*i+r], fontsize=18, fontweight='bold')
            
    
        # Set colorbar
        cbaxes = fig.add_axes([0.135, bar_y[i], 0.75, 0.016]) #left, bottom, width, height
        cbar = fig.colorbar(im, orientation='horizontal', cax=cbaxes,
                            extend='both',
                            shrink=0.6,
                            ticks=levels[i]
                            )
        cbar.set_label(bar_label[i], fontdict=font)
        
    fig.subplots_adjust(left=0.06,top=0.97,bottom=0.09,right=0.96, hspace=0.2, wspace=0.07)
    
    if c == 0:
        fname = './Figures/Figure8_ERA5_all_models_6plots.jpeg'
    elif c == 1:
        fname = './Figures/FigureS8_ERA5_ACCESS-ESM1-5_6plots.jpeg'
    
    fig.savefig(fname, bbox_inches='tight',dpi=300)