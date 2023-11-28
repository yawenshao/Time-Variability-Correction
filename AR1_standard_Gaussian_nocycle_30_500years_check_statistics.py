'''
Script to calculate metrics for emulated 30-year and 500-year time series

Author: Yawen Shao, created on May 30, 2023
'''

import numpy as np
from datetime import date
from statistics_tool import statistics


if __name__ == '__main__':
    # length of time series
    SY = 2148 #if 500 years: 2148; if 30 years: 1678
    EY = 2178
    L = (date(EY, 1, 1)-date(SY, 1, 1)).days
    dd = np.linspace(1,L,L)
    
    
    # model mean temperatures - Gaussian AR1 process -  AR1 normally distributed with a mean of zero + time mean
    mean = 0.0
    AR_var = 1.0
    series = [['low',0.8],['high',0.9]] #make high towards low correlation 
    
    ##### Model high-correlated model series
    corr_high = series[0][1]
    
    ###### Model and correct low-correlated model series
    itr = 100
    corr_low = series[1][1]
    lag1 = []
    lag5 = []
    wsdi = []
    lag1_obs = []
    lag5_obs = []
    wsdi_obs = []
    lag1_corr = []
    lag5_corr = []
    wsdi_corr = []
    
    for m in range(itr):
        Q = (1.0 - pow(corr_high, 2)) * AR_var # #3.5 
        
        Tt = np.full_like(dd, np.nan)
        rands = np.random.normal(0, np.sqrt(Q), L)
        Tt[0] = rands[0]
        
        for i in range(1,L):
            Tt[i] = Tt[i-1] * corr_high + rands[i]
        Tt += mean
        
        Q = (1.0 - pow(corr_low, 2)) * AR_var # #3.5 
            
        Tt2 = np.full_like(dd, np.nan)
        rands = np.random.normal(0, np.sqrt(Q), L)
        Tt2[0] = rands[0]
        
        for i in range(1,L):
            Tt2[i] = Tt2[i-1] * corr_low + rands[i]
        Tt2 += mean
    
        ## Improve series' lag-1 correlation for poorly correlated series
        Tt_c = np.full_like(Tt2, np.nan)
        Tt_c[0] = Tt2[0]
        
        for i in range(1,L):
            Tt_c[i] = Tt_c[i-1] * corr_high + np.sqrt(1.0-pow(corr_high,2))*(Tt2[i]-corr_low*Tt2[i-1])/np.sqrt(1.0-pow(corr_low,2))
    
        Tt_cc = np.mean(Tt) + (Tt_c-np.mean(Tt_c))*np.sqrt(np.var(Tt)/np.var(Tt_c))
        
        
        stat_model = statistics(Tt2)
        lag1.append(stat_model.lag_corr_nonan_notr_30y_mv(1, 1))
        lag5.append(stat_model.lag_corr_nonan_notr_30y_mv(5, 1))
        prcts = stat_model.base_period_percentile(Tt2, SY, EY-1, 5)
        wsdis = stat_model.WSDI(6, False, prcts, SY, EY-1)
        wsdi.append(np.nanmean(wsdis))
    
        stat_model = statistics(Tt)
        lag1_obs.append(stat_model.lag_corr_nonan_notr_30y_mv(1, 1))
        lag5_obs.append(stat_model.lag_corr_nonan_notr_30y_mv(5, 1))
        prcts = stat_model.base_period_percentile(Tt, SY, EY-1, 5)
        wsdis = stat_model.WSDI(6, False, prcts, SY, EY-1)
        wsdi_obs.append(np.nanmean(wsdis))
        
        stat_model = statistics(Tt_cc)
        lag1_corr.append(stat_model.lag_corr_nonan_notr_30y_mv(1, 1))
        lag5_corr.append(stat_model.lag_corr_nonan_notr_30y_mv(5, 1))
        prcts = stat_model.base_period_percentile(Tt_cc, SY, EY-1, 5)
        wsdis = stat_model.WSDI(6, False, prcts, SY, EY-1)
        wsdi_corr.append(np.nanmean(wsdis))
    
    
    np.savetxt('./Data/Raw_new_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_lag1Corr_AR1_'+str(itr)+'.csv', lag1, fmt = '%5.3f', delimiter=",")
    np.savetxt('./Data/Raw_new_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_lag5Corr_AR1_'+str(itr)+'.csv', lag5, fmt = '%5.3f', delimiter=",")
    np.savetxt('./Data/Raw_new_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_wsdi_AR1_'+str(itr)+'.csv', wsdi, fmt = '%5.3f', delimiter=",")
    
    np.savetxt('./Data/Corrected_new_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_lag1Corr_AR1_'+str(itr)+'.csv', lag1_corr, fmt = '%5.3f', delimiter=",")
    np.savetxt('./Data/Corrected_new_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_lag5Corr_AR1_'+str(itr)+'.csv', lag5_corr, fmt = '%5.3f', delimiter=",")
    np.savetxt('./Data/Corrected_new_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_wsdi_AR1_'+str(itr)+'.csv', wsdi_corr, fmt = '%5.3f', delimiter=",")
    
    np.savetxt('./Data/Obs_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_lag1Corr_AR1_'+str(itr)+'.csv', lag1_obs, fmt = '%5.3f', delimiter=",")
    np.savetxt('./Data/Obs_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_lag5Corr_AR1_'+str(itr)+'.csv', lag5_obs, fmt = '%5.3f', delimiter=",")
    np.savetxt('./Data/Obs_'+str(EY-SY)+'y_'+str(corr_high)+'_'+str(corr_low)+'_wsdi_AR1_'+str(itr)+'.csv', wsdi_obs, fmt = '%5.3f', delimiter=",")
