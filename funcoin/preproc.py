import numpy as np


"""
Preprocessing functions for Functional Connectivity Integrative Normative Modelling (FUNCOIN)
@Author and maintainer of Python package: Janus Rønn Lind Kobbersmed, janus@cfin.au.dk or januslind@gmail.com
@Based on the Covariate-Assisted Principal regression method: Zhao, Y. et al. (2021). 'Covariate Assisted Principal regression for covariance matrix outcomes', Biostatistics, 22(3), pp. 629-45.  
"""

def standardise_ts(Y_dat, standard_var=True):
    """Standardises the time series data to mean 0 (regionwise) and variance 1 (optional).
        
    Parameters:
    -----------
    Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series. 
    standard_var (optional): If True, the time series data is standardized to variance 1 (region-wise).
    
    Returns:
    --------
    Y_stand: List of length [number of subjects] containing standardized time series data, where each element is a matrix of shape [no. of time points]x[no. of regions]. 
    """    

    n_subj = len(Y_dat)

    Y_demeaned = [Y_dat[i] - np.mean(Y_dat[i], axis=1) for i in range(n_subj)]
    
    if standard_var:
        Y_stand = [Y_demeaned[i]/np.std(Y_demeaned[i], axis=1) for i in range(n_subj)]
    else:
        Y_stand = Y_demeaned

    return Y_stand