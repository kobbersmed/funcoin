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
    Y_demeaned = [Y_dat[i] - np.mean(Y_dat[i], axis=0) for i in range(n_subj)]
    
    if standard_var:
        Y_stand = [Y_demeaned[i]/np.std(Y_demeaned[i], axis=0) for i in range(n_subj)]
    else:
        Y_stand = Y_demeaned

    return Y_stand

def create_TDE(ts, lags, standardise_mean = True, standardise_var = True):
    """ Creates a time-delay embedded version of the time series data from a single subject.
    
    Parameters:
    -----------
    ts: Array of shape (T,p), with T the number of time points and p the number of regions/time series.
    lags: List or arrays of lags
    standardise_mean: Boolean. If true, the mean is subtracted for each channel/region after time delay embedding making the mean of each channel equal to 0.
    standardise_mean: Boolean. If true, the time series from each channel/region is divided by its standard deviation after time delay embedding making the variance equal to 1. 

    Returns:
    --------
    ts_TDE: Array of shape (T-L, L*p), with T the number of time points, L the number of non-zero lags, and p the number of regions/time series
    """""

    T,p = ts.shape

    L = len(lags)

    ts_TDE_init = np.zeros([T, p*L])

    start_ind = np.abs(np.minimum(np.min(lags),0))
    end_ind = T-np.maximum(np.max(lags), 0)

    for i1 in range(p):
        for i2 in range(L):
            ts_TDE_init[:, i1*L + i2] = np.roll(ts[:,i1], lags[i2], axis=0)

    ts_TDE = ts_TDE_init[start_ind:end_ind,:]

    if standardise_mean:
        ts_TDE  = ts_TDE - np.mean(ts_TDE,0)

    if standardise_var:
        ts_TDE  = ts_TDE / np.std(ts_TDE,0)
        
    return ts_TDE

def create_FC_TDE(ts, lags, cov_type = 'Pearson', ddof = 0):
    """Creates the functional connectivity (FC) matrix of time delay embedded data (TDE) without actually making the full TDE data as an intermediate step.
        
    Parameters:
    -----------
    ts: Array of shape (T,p) with T being the number of time points and p being the number of regions/signals.
    lags: List of integers. Contains the lag values to be used in the TDE.
    cov_type: String specifying the type of FC matrix to be generated. Either 'Covariance' (covariance matrix) or 'Pearson' (correlation matrix). Default value is 'Pearson').
    ddof: Specifies "delta degrees of freedom" for the input FC matrices. The divisor used for calculating the input FC matrices is T-ddof, with T being the number of time points.
            Unbiased covariance (sample covariance) matrix has ddof = 1, which is default when calling numpy.cov(). Population covariance is calculated with ddof=0. Default value is ddof=0.
    Returns:
    --------
    FC_TDE: Array of shape (p*L, p*L), with p being the number of regions/signals and L being the number of lags. The matrix is the FC matrix of the TDE time series.
    """        

    T, p = ts.shape
    L = len(lags)
    lags_sort = np.sort(np.array(lags))[::-1]
    start_ind = int(np.abs(np.minimum(np.min(lags_sort),0)))
    end_ind = int(T-np.maximum(np.max(lags_sort), 0))
    FC_TDE = np.zeros((p*L,p*L))
    diag_els = np.ones(p*L)
    for i in range(p*L):
        row = []
        chan_ind1 = int(np.floor(int(i/L)))
        lag_ind1 = i % L

        ts1 = ts[(start_ind+lags_sort[lag_ind1]):(end_ind+lags_sort[lag_ind1]), chan_ind1]

        if cov_type == 'Covariance':
            diag_els[i] = np.var(ts1, ddof = ddof) 

        for i2 in range(i+1,p*L):
            chan_ind2 = int(np.floor(int(i2/L)))
            lag_ind2 = i2 % L

            
            ts2 = ts[(start_ind+lags_sort[lag_ind2]):(end_ind+lags_sort[lag_ind2]), chan_ind2]

            if cov_type == 'Pearson':
                corr_val = np.corrcoef(ts1, ts2)[0,1]
            elif cov_type == 'Covariance':
                corr_val = np.cov(ts1, ts2, ddof=ddof)[0,1]

            row.append(corr_val)

        FC_TDE[i,i+1:] = np.array(row)
        row.append(corr_val)

    FC_TDE = FC_TDE + FC_TDE.T + np.diag(diag_els)
    return FC_TDE

def create_symmetric_lags(window_length):
    """ Creates a list of symmetric lag values around 0.
    
    Parameters:
    -----------
    window_length: The length of the window (in number of time points). If window_length is odd, the lags are symmetric around 0. Otherwise, there will be one more negative lag than positive.
    
    Returns:
    --------
    lags: List of indices corresponding to a window of the specified length around the zero-lag. 
    """""

    shift = int(np.floor(window_length/2))
    lags = np.arange(window_length)-shift
    return lags

