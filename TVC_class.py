"""
Class to implement the Time Variability Correction method

Author: Yawen Shao, created on May 30, 2023
"""

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm


class TVC:
    def __init__(self, obs, raw_train, raw_val, scales):
        """
        Initialise global variables in the Time Variability Correction method
        obs - observational data series
        raw_train - raw training model series
        raw_val - raw validation model series
        scales - the smoothers list, e.g. [365,183,92,46,23,12,6,3,2]
        """
        
        # Check data length
        try:
            assert len(obs) == len(raw_train)
        except:
            print('The length of observation and raw data is not the same. Exiting...')
            exit()
        
        # Assign variables
        self.scales = scales
        self.obs = obs
        self.train = raw_train
        self.val = raw_val

    def sep_time_scales_rolling(self, var):
        '''
        The wavelet based transform approach to filter the original time series
        '''
        
        # Determine the variable to be analysed
        if var == 'obs':
            data = self.obs
        elif var == 'train':
            data = self.train
        elif var == 'val':
            data = self.val
        
        # The length of data and level of smoothers
        n = len(data)
        l = len(self.scales)
        
        y_Pkn = np.full((n, l), np.nan)
        y_Pkn_ano = np.full((n, l), np.nan)
        
        data_df = pd.DataFrame(data)
        
        # Filter the time series one by one
        # The Nan values are handled
        for k, smer in enumerate(self.scales): # timescale of evolution
            if k == 0:
                y_Pkn[:,k] = data_df.rolling(smer, center=False, min_periods=1).mean().to_numpy().flatten()
                y_Pkn_ano[:,k] = data - y_Pkn[:,k]
            else:
                y_ano_df = pd.DataFrame(y_Pkn_ano[:,k-1])
                y_Pkn[:,k] = y_ano_df.rolling(smer, center=False, min_periods=1).mean().to_numpy().flatten()
                y_Pkn_ano[:,k] = y_Pkn_ano[:,k-1] - y_Pkn[:,k]
                
            y_Pkn[np.where(np.isnan(y_Pkn_ano[:,k])),k] = np.nan
    
        # Obtain data with filtered time series
        init_ind = sum(self.scales)-l
        L_data = n - init_ind
        
        Y = np.full((L_data,l+1), np.nan)
        Y[:,-1] = y_Pkn_ano[init_ind:,-1]
        Y[:,:-1] = y_Pkn[init_ind:,:]
        
        return Y
    
    def stat_obs(self, data_m):
        '''
        Calculate the mean and covariance of the data matrix
        '''
        # Remove nan values
        filter_data = np.argwhere(np.isnan(data_m[:,0]))
        data_m = np.delete(data_m, filter_data, axis=0)
        
        # Calculate mean vector
        data_mean = np.mean(data_m, axis=0)
            
        # Calculate covariance matrix
        cov_data = (np.transpose(data_m) - data_mean.reshape((len(data_mean),1))) @ (data_m - data_mean)/(data_m.shape[0]-1)
        
        return data_mean, cov_data
    
    def bias_correction(self, X):
        '''
        Correct the target data matrix by defining a new matrix Z
        Y_mean - mean vector of the observation matrix
        Y_cov - covariance matrix of the observation matrix
        X_mean - mean vector of the raw matrix over the training period
        X_cov - covariance matrix of the raw matrix over the training period
        X - the time series matrix to be corrected
        '''
        X_corr = np.full_like(X, np.nan)
        
        # Correct means
        for i in range(X.shape[1]):
            delta_mu = self.Y_mean[i] - self.X_mean[i]
            X_corr[:,i] = X[:,i] + delta_mu
        
        # Define a neew Z matrix
        Z = np.mean(X_corr, axis=0) + (X_corr - np.mean(X_corr, axis=0)) @ sqrtm(np.linalg.inv(self.X_cov)) @ sqrtm(self.Y_cov)
    
        return Z

    def TVC_postprocess(self):
        '''
        Post-processing by Time Variability Correction method
        '''

        ## Step 1: model fitting
        Y_data = self.sep_time_scales_rolling('obs')
        self.Y_mean, self.Y_cov = self.stat_obs(Y_data)
		
        X_data = self.sep_time_scales_rolling('train')
        self.X_mean, self.X_cov = self.stat_obs(X_data)	
		
        ## Step 2: model validation
        # Apply to validation period
        X_val = self.sep_time_scales_rolling('val')

        ## Step 3: Obtain TVC post-pocessed data matrix
        Z = self.bias_correction(X_val)

        return Z

    def get_TVC_covariance(self, var):
        '''
        To obtain the covariance matrix of target time series
        '''
        data = self.sep_time_scales_rolling(var)
        _, cov = self.stat_obs(data)
		
        return cov

    def get_decomposition(self, var):
        '''
        To obtain the data matrix of filtered time series
        '''
        data = self.sep_time_scales_rolling(var)
		
        return data