"""
Class to compute verification metrics

Author: Yawen Shao, created on May 30, 2023
"""

import numpy as np
import pandas as pd
import calendar
import operator
from numpy.lib.stride_tricks import sliding_window_view

class statistics:
    def __init__(self, data):
        """
        Initialise global variables in the class
        data - data series to be analysed
        """
        
        self.data = data
        
    def nday_discrete_average_df(self, n):
        '''
        Average of n-day data with length of len(data)//2
        '''

        data_df = pd.DataFrame(self.data)

        series = data_df.rolling(n, center=False, min_periods=1).mean().to_numpy().flatten()
        
        return series[n-1:][::n]

    def lag_corr_nonan_notr_30y_mv(self, win, lag):
        '''
        Lag correlations relative to 30-year running mean
        '''
        
        if win == 1:
            data = self.data
        else:
            data = self.nday_discrete_average_df(win)
        
        data_df = pd.DataFrame(data)
        
        lens = 10957//win # length of 30 years

        data_mean = data_df.rolling(lens, center=True, min_periods=lens).mean()
        data_mean = data_mean.to_numpy().flatten()

        data_mean[:lens//2] = data_mean[lens//2]
        data_mean[-lens//2:] = data_mean[-lens//2]

        up = np.mean((data[lag:]-data_mean[lag:])*(data[:-lag]-data_mean[:-lag]))
        down = np.mean(np.power(data-data_mean,2))

        corr_trend = up/down

        return corr_trend

    def sample_var_30y_mv(self):
        '''
        Sample variance relative to 30-year running mean
        '''
        
        data_df = pd.DataFrame(self.data)

        lens = 10957 # length of 30 years
        data_mean = data_df.rolling(lens, center=True, min_periods=lens).mean()
        data_mean = data_mean.to_numpy().flatten()

        data_mean[:lens//2] = data_mean[lens//2]
        data_mean[-lens//2:] = data_mean[-lens//2]

        variance = np.sum(np.power(self.data - data_mean, 2))/(len(self.data)-1)
        
        return variance

    def WSDI(self, nday, span_yr, prct, yr_st, yr_end):
        '''
        The warm spell duration index is calculated following the general procedures
        used by the Expert Team on Climate Change Detection and Indices (ETCCDI) 
        Definition: annual count of days with at least n consecutive days when daily Tmax > 90th percentile
        span_yr - whether allow WSDI to span between years
        nday - number of consecutive days
        yr_st - start year of the data series
        yr_end - end year of the data series
        output - warm spell duration index for each year

        Climpact permits a maximum of 3 missing days in any month, and a maximum of 15 missing days in any year. 
        If any of these thresholds is exceeded then the month or year in question is not calculated. 
        '''
        
        self.dates = pd.date_range(str(yr_st)+'-01-01',str(yr_end)+'-12-31')
        self.yr_st = yr_st
        self.yr_end = yr_end
        self.data = self.data[-len(self.dates):]
        
        # Check the data quality, whether include nan values
        self.check_input_years()
        
        wsdi_all = self.move_window_prct_op(operator.gt, nday, span_yr, prct)
        
        # Sum the annual consecutive days
        data_df = wsdi_all.groupby(wsdi_all.index.year).sum()
        
        data_final = data_df.to_numpy().flatten()

        if self.year_mis:
            for j in self.year_mis:
                data_final[int(j)-self.yr_st] = np.nan
            
        return data_final
    
    def check_input_years(self):
        '''
        Check whether and how many nan values input data include
        '''
        
        ## Find years with 1) have > 15d N/A in a year 2) have > 3d N/A in a month
        data_df = pd.DataFrame(self.data, index=self.dates)
        
        # 1) have > 15d N/A in a year
        data_y = data_df.apply(lambda x: pd.isna(x)).groupby(data_df.index.year).sum()
        data_y = data_y.index[data_y.iloc[:,0] >= 15].tolist()
        
        # 2) have > 3d N/A in a month
        data_m = data_df.apply(lambda x: pd.isna(x)).groupby([data_df.index.year, data_df.index.month]).sum()
        data_m = data_m.reset_index(level=1)
        data_m = data_m.index[data_m.iloc[:,1] >= 3].tolist()
        
        self.year_mis = list(set(data_y) | set(data_m))
        
        return
    
    def base_period_percentile(self, base_data, base_st, base_end, win_len):
        '''
        Base period data preparation
        Calculate the 90th percentiles for each calendar day
        base_data - base period data
        base_st - start year of the base period
        base_end - end year of the base period
        win_len - length of the moving window
        '''
        
        base_dates = pd.date_range(str(base_st)+'-01-01', str(base_end)+'-12-31')
        ## Step1: remove data on Feb 29
        ind = np.where((base_dates.month==2) & (base_dates.day==29))
        data_365 = np.delete(base_data, ind)
        
        ## Step2: create a n*window_length matrix
        data_nday = sliding_window_view(data_365, window_shape = win_len)
        
        c = win_len//2
        # First c rows
        firsts = sliding_window_view(np.concatenate((data_365[-c:], data_365[:win_len-1])), window_shape = win_len)
        # End c rows
        ends = sliding_window_view(np.concatenate((data_365[-(win_len-1):], data_365[:c])), window_shape = win_len)
        
        ## Final matrix array
        data_nday = np.vstack((firsts, data_nday, ends))
        data_nday = data_nday.reshape((base_st-base_end+1, 365, win_len))
        
        # Calculate percentiles
        prcts = np.nanpercentile(data_nday, 90, method='median_unbiased', axis=[0, 2])
        
        return prcts

    def move_window_prct_op(self, op, nday, span_yr, prcts):
        '''
        Flag the days counted in the warm spell duration index
        '''
        
        data_prct_all = []
        
        data_prct_366 = np.insert(prcts, 59, prcts[58]).tolist()
        ### leap yar - repeat 2/28 to 2/29
        for year in range(self.yr_st, self.yr_end+1):
            if calendar.isleap(year):
                data_prct_all += data_prct_366
            else:
                data_prct_all += prcts.tolist()

        data_pf = pd.DataFrame(self.data, index=self.dates)
        data_f = op(data_pf.iloc[:,0], data_prct_all)
        
        # Determine n-day consecutive days
        # Whether consider consecutive days spanning across two years
        data_flag = self.consecutive_day_filter(data_f, nday, span_yr)
        
        return data_flag

    def consecutive_day_filter(self, data_f, nday, span_yr):
        '''
        Flag the consecutive days satisfying the condition
        '''
        
        if span_yr:
            data_flag = data_f.rolling(nday, center=False, min_periods=1).sum()
        else: # Only consider consecutive days in a year
            data_flag = data_f.groupby(data_f.index.year).rolling(nday, center=False, min_periods=1).sum().reset_index(drop=True)

        ind_nday = np.where(data_flag == nday)[0]
        
        for i in range(1, nday):
            data_flag.iloc[ind_nday-i] = nday
                
        data_flag = data_flag.where(data_flag >= nday, np.nan)
        
        data_flag_final = data_flag.where(pd.isna(data_flag), 1)
        data_flag_final = pd.DataFrame(data_flag_final.to_numpy(), index=self.dates) 

        return data_flag_final