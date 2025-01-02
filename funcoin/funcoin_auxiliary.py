import numpy as np
import warnings

def calc_covmatrix_listtolist(Y_dat, ddof=0):
    """Calculates the Pearson correlation matrix, which is used to find the transformed data in the FUNCOIN class method transform_timeseries.
    In the FUNCOIN model, time series data is standardized (mean 0, variance 1) and the covariance matrix with n degrees of freedom is computed,
    which is equivalent.
    
    Parameters:
    Y_dat: List of length n_subj with time series data for each subject. Each element of the list should be array-like of shape (n_timepoints, n_regions)
    ddof: Integer. Delta degrees of freedom.
    ###
    """

    covmat_list = [np.cov(Y_dat[i], ddof = ddof, rowvar=False) for i in range(len(Y_dat))] 

    return covmat_list

def test_matrixdef(mat):
    """Tests definitenes of a real (symmetric) matrix
    ----------
    Parameters:
      mat : array-like of shape (p,p)
    Outputs:
      def_stat: -1 if negative definite, 1 if positive definite, -0.1 if SND, 0.1 if SPD, 0 if indefinite
    ----------

    """

    sym_stat = test_matrixsym(mat)

    if not sym_stat:
        print("Warning: Matrix is not symmetric")
        return float('NaN')

    eigvals = np.linalg.eigvals(mat)
    n_eigs = len(eigvals)

    if sum(eigvals>0) == n_eigs:
        def_stat = 1
    elif sum(eigvals >= 0) == n_eigs:
        def_stat = 0.1
    elif sum(eigvals < 0) == n_eigs:
        def_stat = -1
    elif sum(eigvals <= 0) == n_eigs:
        def_stat = -0.1
    else:
        def_stat = 0

    return def_stat


def test_matrixsym(mat):
    #Tests symmetry of a square matrix
    #Parameters:
    #   mat: array-like of shape pxp
    # 
    #Returns:
    #   sym_stat: True if symmetric, False if non-symmetric
    
    sym_stat = np.all(abs(mat-mat.T)<1e-6)
    
    return sym_stat

def dfd_func(matA):

    test_def = test_matrixdef(matA)

    if test_def != 1:
        warnings.warn('Matrix is not PD. DFD measure is only defined for PD matrices')

    dfd_val = np.prod(np.diag(matA))/np.linalg.det(matA)

    return dfd_val

def make_Si_list(Y_dat):
    """Creates list of Si matrices from the time series data of each subject. Si is the correlation matrix of subject i without dividing by the number of time points, i.e the sum of y@y.T at each time point (i.e. the scatter matric).

    Parameters:
    ----------- 
    Y_dat: List of length [number of subjects] contatining (demaned and variance 1) time series data for each subject. Each element of the list should be array-like of shape (n_timepoints, n_regions).

    Returns:
    --------
    Si_list: List of matrices of size (p,p) for each subject. Si is scatter matrix of subject i.
    """

    Si_list = [sum([np.outer(Y_dat[i][k,:],Y_dat[i][k,:]) for k in range(Y_dat[i].shape[0])]) for i in range(len(Y_dat))]
    return Si_list

def make_Xi_list(X_dat):
    """Creates a list of vectors of size (q,1) containing the covariates of each subject.

    Parameters:
    ----------- 
    X_dat: Array of size (n_subj,q). Each 

    Returns:
    --------
    Xi_list: List of arrays of size (q,1) for each subject.
    """

    Xi_list = [np.expand_dims(X_dat[i,:],1) for i in range(X_dat.shape[0])]
    return Xi_list


