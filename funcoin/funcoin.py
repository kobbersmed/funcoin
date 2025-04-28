#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functional Connectivity Integrative Normative Modelling (FUNCOIN)
@Author and maintainer of Python package: Janus Rønn Lind Kobbersmed, janus@cfin.au.dk or januslind@gmail.com
@Based on the Covariate-Assisted Principal regression method: Zhao, Y. et al. (2021). 'Covariate Assisted Principal regression for covariance matrix outcomes', Biostatistics, 22(3), pp. 629-45.  
""" 

import numpy as np
import warnings
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from . import funcoin_auxiliary as fca
import importlib

class Funcoin:
    """
    Class for Functional Connectivity Integrative Normative Modelling (FUNCOIN).

    Attributes:
    -----------
    gamma: False or array-like of shape (q, n_comps). If provided, creates the FUNCOIN class with a predefined Gamma matrix. Default value False.
    beta: False or array-like of shape (n_covariates x n_comps). If provided, creates the FUNCOIN class with a predefined Beta matrix. Default value False.
    dfd_values_training: NaN or vector of size [number of projections] containing "deviation from diagonality" values for the data used to train the model (i.e. identify the projections).
                         The attribute is automatically defined when training the model. 
    residual_std_train: NaN or array of length [no. of components]. Element j is the standard deviation of the residuals for the transformed values of projection j. 
    beta_bootstrap: Nan or list of length [number of bootstrap samples] containing beta matrices from the bootstrapping procedure.
                         The attribute is defined when running the method .decompose_bootstrap().
    beta_CI_bootstrap: Nan or list of length 2 containing matrices whose elements are the lower and upper bounds of the elementwise confidence intervals of the specified confidence level for the beta matrix.
                        These are determined from the bootstrapping procedure.
    beta_CI95_parametric: Nan or list of length 2 containing matrices whose elements are the lower and upper bounds of the elementwise confidence intervals for the beta matrix.
                        Only non-Nan if fitted with betaLinReg set to true. The limits are determined from the SE of beta coefficients identified with linear regression.
    u_training: Nan or array-like of shape n_subj x [number of projections]. Contains the transformed data values (u values) of the data the model was trained on.  
    decomp_settings: Python dictionary. Stores variables defined (manually or by default) when calling the method .decompose. This includes: max_comps, gamma_init, rand_init, n_init, max_iter, tol, trace_sol, seed, betaLinReg
                    For details, see the docstring of the decompose method.
    gamma_steps_all, beta_steps_all: List of length [no. of projections]. Element j contains a list of length [no. of iterations for projection j] containing the steps in the optimization algorithm. Only the trace from the initial condition giving the best fit is kept.
    __fitted: Private variable, which is False per default and set to True only if the model is fitted on data (i.e. if gamma and beta are not predefined). Accessed by calling the class method .isfitted(). 
    """

    def __init__(self, gamma=False, beta=False):
        """Constructs relevant instance variables as either False (__fitted), predefined (gamma or beta), or NaN (all others).
        """
        self.gamma = gamma
        self.beta = beta
        self.dfd_values_training = float('NaN')
        self.residual_std_train = float('NaN')
        self.betas_bootstrap = float('NaN')
        self.beta_CI_bootstrap = float('NaN')
        self.beta_CI95_parametric = float('NaN')
        self.u_training = float('NaN')
        self.decomp_settings = dict()
        self.gamma_steps_all = []
        self.beta_steps_all = []
        self.__fitted = False

    def __str__(self):
        firststr = 'Instance of the Functional Connectivity Integrative Normative Modelling (FUNCOIN) class. '

        if self.__fitted:
            fitstr = 'have been fitted.'
        else:
            fitstr = 'are predefined.'

        if (self.gamma is False) and (self.beta is False):
            laststr = f'Neither gamma nor beta are defined.'
        elif (self.gamma is not False) and (self.beta is False):
            laststr = f'gamma is predefined. beta is not defined.'
        elif (self.gamma is False) and (self.beta is not False):
            laststr = f'beta is predefined. gamma is not defined.'
        elif (self.gamma is not False) and (self.beta is not False):
            laststr = f'gamma and beta ' + fitstr

        return firststr + laststr

    #Public methods
    def decompose(self, Y_dat, X_dat, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, trace_sol = 0, seed_initial = None, betaLinReg = True, overwrite_fit = False, *kwargs):
        """Performs FUNCOIN decomposition given a list of time series data, Y_dat, and covariate matrix, X_dat. 
        
        Parameters:
        -----------
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series. 
        X_dat: Array-like of shape (n_subjects, q). First column has to be ones (does not work without the intercept).
        max_comps: Maximal number of components (gamma), to be identified. May terminate with fewer components if a singular matrix occurs during the optimisation. This may happen 
                    if the problem is ill-posed, e.g. if no common components can be found or the common directions of variance have already been identified.
        gamma_init: False or array of length n_regions. If not False, the optimization algorithm uses the array as initial condition for gamma. In the optimisation algorithm, beta is determined from the current gamma, 
                    without utilising an initial beta matrix. Default value False.
        rand_init: Boolean. If True, the decomposition will use random initial gamma values. Default value True.
        n_init: Integer. Default value 20. 
        max_iter: Integer>0. Maximal number of iterations when determining gamma (and beta, if betaLinReg is False). Default value 1000.
        tol: Float >0. Maximal tolerance when optimizing for gamma (and beta). If an iteration yields an absolute change smaller than this value in all elements of gamma and beta, the optimisation stops. Default 1e-4.
        trace_sol: Boolean. Whether or not to keep all intermediate steps in the optimization of gamma and beta. The steps are stored in lists in instance variables self.gamma_steps_all and self.beta_steps_all.
        seed_initial: Integer or None. If integer, this seeds the random initial conditions. Default value False.
        betaLinReg: Boolean. If true, the algorithm concludes with performing ordinary linear regression on the transformed values using the gamma transformation found to improve accuracy of beta estimation. Default False.
        overwrite_fit: Boolean. If False: Returns an exception if the class object has already been fitted to data. If True: Fits using the provided data and overwrites any existing values of gamma, beat, dfd_values_training, and u_training.

        Returns:
        --------
        self.beta: Array-like of shape (q,n_dir). Coefficients of the log-linear model identified during decomposition.
        self.gamma: Array-like of shape (p,n_dir). Matrix with each column being an identified gamma projection.
        self.dfd_values_training: Array of length n_dir. Contains the average values of "deviation from diagonality" computed on the data used to fit the model. This can be used for selecting the number of projections (see Zhao, Y. et al. (2021)). 
        self.u_training: The transformed values of the data used to fit the model, i.e. the logarithm of the diagonal elements of Gamma.T @ Sigma_i @Gamma for subject i.
        self.residual_std_train: Projection-wise standard deviation of the residuals (i.e. transformed values minus the mean). Computed assuming homogeneity of variance.
        self.beta_CI95_parametric: Nan or list of length 2 containing matrices whose elements are the lower and upper bounds of the elementwise confidence intervals for the beta matrix.
                        Only non-Nan if fitted with betaLinReg set to true. The limits are determined from the SE of beta coefficients identified with linear regression.
        self.beta_pvals: Nan or array-like of shape (q,n_dir). If non-Nan, the array contains the coefficient-wise p-values of the hypothesis test for the beta coefficient being equal to 0. Significance level is 0.05. 
                        Only non-Nan if fitted with betaLinReg set to true. The p-values are determined from coefficient-wise t-tests of beta coefficients identified with linear regression.
        self.beta_tvals: Nan or array-like of shape (q,n_dir). If non-Nan, the array contains the coefficient-wise t-values of the hypothesis test for the beta coefficient being equal to 0.
                        Only non-Nan if fitted with betaLinReg set to true. The t-values are determined from the SE of beta coefficients identified with linear regression.
        self.decomp_settings: Dictionary. When running the decomposition method, settings are stored in this dictionary (e.g. number of components, initial conditions, number of iterations, tolerance, etc.) 
                        
        Raises:
        -------
        Exception: Raises exception if the model has already been fitted and overwrite_fit is False.
        Exception: Raises exception if no common components (gammas) can be identified.
        Exception: Raises and handles exception, if a singular matrix occurs during the optimisation. This may happen if the problem is ill-posed, e.g. if no common components can be found or the common directions of 
                    variance have already been identified. Upon this exception, the gamma and beta already identified are kept.
        """

        self.decomp_settings['max_comps'] = max_comps
        self.decomp_settings['gamma_init'] = gamma_init
        self.decomp_settings['rand_init'] = rand_init
        self.decomp_settings['n_init'] = n_init
        self.decomp_settings['max_iter'] = max_iter
        self.decomp_settings['tol'] = tol
        self.decomp_settings['trace_sol'] = trace_sol
        self.decomp_settings['seed_initial'] = seed_initial
        self.decomp_settings['betaLinReg'] = betaLinReg


        isfit = self.isfitted()

        try:
            add_to_fit = kwargs['add_to_fit']
        except:
            add_to_fit = False

        if isfit and not (overwrite_fit or add_to_fit):
            raise Exception('Did not run the decomposition, because this FUNCOIN instance has already been fitted. If you want to overwrite existing fit, please specify overwrite_fit=True. If you want to add more projection directions to the existing fit, use the .add_projections method.')
        if not isfit and np.ndim(self.gamma)>0 and not add_to_fit:
            warnings.warn('Fitting FUNCOIN instance. Predefined gamma and beta will be overwritten.')
        if overwrite_fit and np.ndim(self.gamma)>0:
            warnings.warn('Running FUNCOIN decomposition. Overwriting existing fit.')

        gamma_mat, beta_mat = self.__decomposition(Y_dat, X_dat, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed = seed_initial, betaLinReg = betaLinReg, overwrite_fit=overwrite_fit, add_to_fit=add_to_fit)

        self.__fitted = True

        self.gamma = gamma_mat
        self.beta = beta_mat

        u_vals_training = self.transform_timeseries(Y_dat)
        self.u_training = u_vals_training

        model_pred_training = X_dat@beta_mat
        self.residual_std_train = np.std(u_vals_training-model_pred_training, axis=0, ddof = 1)

        Ti_vec = [Y_dat[i].shape[0] for i in range(len(Y_dat))]
        Ti_equal = np.all([Ti_vec[i]==Ti_vec[0] for i in range(len(Ti_vec))])

        w_io = not Ti_equal

        dfd_values_training = self.calc_dfd_values(Y_dat, weighted_io=w_io, dfd_aritm = 0, logtrick_io = 1)
        self.dfd_values_training = dfd_values_training

        if betaLinReg:
            beta_CI95_parametric = self.CI_beta_parametric(X_dat=X_dat, CI_lvl=0.05)
            self.beta_CI95_parametric = beta_CI95_parametric
            beta_pvals, beta_tvals = self.ttest_beta(X_dat=X_dat, h0_beta=False)
            self.beta_pvals = beta_pvals
            self.beta_tvals = beta_tvals

    def decompose_FC(self, FC_list, X_dat, Ti_list, ddof = 0, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, trace_sol = 0, seed_initial = None, betaLinReg = True, overwrite_fit = False, *kwargs):
        """Performs FUNCOIN decomposition given a list of FC matrices, FC_list, and covariate matrix, X_dat. 
        
        Parameters:
        -----------
        FC_list: List of length [number of subjects] containing covariance/correlation matrix for each subject. Each element of the list should be array-like of shape (p, p), with p the number of regions. 
        X_dat: Array-like of shape (n_subjects, q). First column has to be ones (does not work without the intercept).
        Ti_list: Int or list of length [number of subjects] with integer elements. The number of time points in the time series data for all/each subject(s). If the elements are not equal, a weighted average deviation from diagonality values is computed. If all
                    subjects have the same number of time points, this can be specified with an integer instead of a list of integers.
        ddof: Specifies "delta degrees of freedom" for the input FC matrices. The divisor used for calculating the input FC matrices is T-ddof, with T being the number of time points. Here, default value is 0, which is true for Pearson correlation matrices. 
                    Unbiased covariance (sample covariance) matrix has ddof = 1, which is default when calling numpy.cov(). Population covariance is calculated with ddof=0.  
        max_comps: Maximal number of components (gamma), to be identified. May terminate with fewer components if a singular matrix occurs during the optimisation. This may happen 
                    if the problem is ill-posed, e.g. if no common components can be found or the common directions of variance have already been identified.
        gamma_init: False or array of length n_regions. If not False, the optimization algorithm uses the array as initial condition for gamma. In the optimisation algorithm, beta is determined from the current gamma, 
                    without utilising an initial beta matrix. Default value False.
        rand_init: Boolean. If True, the decomposition will use random initial gamma values. Default value True.
        n_init: Integer. Default value 20. 
        max_iter: Integer>0. Maximal number of iterations when determining gamma (and beta, if betaLinReg is False). Default value 1000.
        tol: Float >0. Maximal tolerance when optimizing for gamma (and beta). If an iteration yields an absolute change smaller than this value in all elements of gamma and beta, the optimisation stops. Default 1e-4.
        trace_sol: Boolean. Whether or not to keep all intermediate steps in the optimization of gamma and beta. The steps are stored in lists in instance variables self.gamma_steps_all and self.beta_steps_all.
        seed_initial: Integer or None. If integer, this seeds the random initial conditions. Default value False.
        betaLinReg: Boolean. If true, the algorithm concludes with performing ordinary linear regression on the transformed values using the gamma transformation found to improve accuracy of beta estimation. Default False.
        overwrite_fit: Boolean. If False: Returns an exception if the class object has already been fitted to data. If True: Fits using the provided data and overwrites any existing values of gamma, beat, dfd_values_training, and u_training.

        Returns:
        --------
        self.beta: Array-like of shape (q,n_dir). Coefficients of the log-linear model identified during decomposition.
        self.gamma: Array-like of shape (p,n_dir). Matrix with each column being an identified gamma projection.
        self.dfd_values_training: Array of length n_dir. Contains the average values of "deviation from diagonality" computed on the data used to fit the model. This can be used for selecting the number of projections (see Zhao, Y. et al. (2021)). 
        self.u_training: The transformed values of the data used to fit the model, i.e. the logarithm of the diagonal elements of Gamma.T @ Sigma_i @Gamma for subject i.
        self.residual_std_train: Projection-wise standard deviation of the residuals (i.e. transformed values minus the mean). Computed assuming homogeneity of variance.
        self.beta_CI95_parametric: Nan or list of length 2 containing matrices whose elements are the lower and upper bounds of the elementwise confidence intervals for the beta matrix.
                        Only non-Nan if fitted with betaLinReg set to true. The limits are determined from the SE of beta coefficients identified with linear regression.
        self.beta_pvals: Nan or array-like of shape (q,n_dir). If non-Nan, the array contains the coefficient-wise p-values of the hypothesis test for the beta coefficient being equal to 0. Significance level is 0.05. 
                        Only non-Nan if fitted with betaLinReg set to true. The p-values are determined from coefficient-wise t-tests of beta coefficients identified with linear regression.
        self.beta_tvals: Nan or array-like of shape (q,n_dir). If non-Nan, the array contains the coefficient-wise t-values of the hypothesis test for the beta coefficient being equal to 0.
                        Only non-Nan if fitted with betaLinReg set to true. The t-values are determined from the SE of beta coefficients identified with linear regression.
        self.decomp_settings: Dictionary. When running the decomposition method, settings are stored in this dictionary (e.g. number of components, initial conditions, number of iterations, tolerance, etc.) 
                        
        Raises:
        -------
        Exception: Raises exception if a non-empty Ti_list is input and FC_list and Ti_list are of unequal lengths.
        Exception: Raises exception if the model has already been fitted and overwrite_fit is False.
        Exception: Raises exception if no common components (gammas) can be identified.
        Exception: Raises and handles exception, if a singular matrix occurs during the optimisation. This may happen if the problem is ill-posed, e.g. if no common components can be found or the common directions of 
                    variance have already been identified. Upon this exception, the gamma and beta already identified are kept.
        """

        if type(Ti_list) == int:
            Ti_val = Ti_list
            Ti_list = [Ti_val for i in range(len(FC_list))]
            Ti_equal = True
        elif type(Ti_list) == list:
            if len(Ti_list)!=len(FC_list):
                raise Exception('Length of list of FC matrices and list of number of time points do not match')
            else:
                Ti_equal = np.all([Ti_list[i]==Ti_list[0] for i in range(len(Ti_list))])


        self.decomp_settings['max_comps'] = max_comps
        self.decomp_settings['gamma_init'] = gamma_init
        self.decomp_settings['rand_init'] = rand_init
        self.decomp_settings['n_init'] = n_init
        self.decomp_settings['max_iter'] = max_iter
        self.decomp_settings['tol'] = tol
        self.decomp_settings['trace_sol'] = trace_sol
        self.decomp_settings['seed_initial'] = seed_initial
        self.decomp_settings['betaLinReg'] = betaLinReg


        isfit = self.isfitted()

        try:
            add_to_fit = kwargs['add_to_fit']
        except:
            add_to_fit = False

        if isfit and not (overwrite_fit or add_to_fit):
            raise Exception('Did not run the decomposition, because this FUNCOIN instance has already been fitted. If you want to overwrite existing fit, please specify overwrite_fit=True. If you want to add more projection directions to the existing fit, use the .add_projections method.')
        if not isfit and np.ndim(self.gamma)>0 and not add_to_fit:
            warnings.warn('Fitting FUNCOIN instance. Predefined gamma and beta will be overwritten.')
        if overwrite_fit and np.ndim(self.gamma)>0:
            warnings.warn('Running FUNCOIN decomposition. Overwriting existing fit.')

        gamma_mat, beta_mat = self.__decomposition(FC_list, X_dat, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed = seed_initial, betaLinReg = betaLinReg, overwrite_fit=overwrite_fit, add_to_fit=add_to_fit, FC_mode=True, Ti_list=Ti_list, ddof = ddof)

        self.__fitted = True

        self.gamma = gamma_mat
        self.beta = beta_mat

        u_vals_training = self.transform_FC(FC_list)
        self.u_training = u_vals_training

        model_pred_training = X_dat@beta_mat
        self.residual_std_train = np.std(u_vals_training-model_pred_training, axis=0, ddof = 1)

        w_io = not Ti_equal

        dfd_values_training = self.calc_dfd_values_FC(FC_list, weighted_io=w_io, dfd_aritm = 0, logtrick_io = 1, Ti_list=Ti_list)
        self.dfd_values_training = dfd_values_training

        if betaLinReg:
            beta_CI95_parametric = self.CI_beta_parametric(X_dat=X_dat, CI_lvl=0.05)
            self.beta_CI95_parametric = beta_CI95_parametric
            beta_pvals, beta_tvals = self.ttest_beta(X_dat=X_dat, h0_beta=False)
            self.beta_pvals = beta_pvals
            self.beta_tvals = beta_tvals

    def decompose_ts(self, Y_dat, X_dat, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, trace_sol = 0, seed_initial = None, betaLinReg = True, overwrite_fit = False, *kwargs):
        """Performs FUNCOIN decomposition given a list of time series data, Y_dat, and covariate matrix, X_dat. This function calls the public method .decompose(), which performs deomposition on time series level.  
        
        Parameters:
        -----------
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series. 
        X_dat: Array-like of shape (n_subjects, q). First column has to be ones (does not work without the intercept).
        max_comps: Maximal number of components (gamma), to be identified. May terminate with fewer components if a singular matrix occurs during the optimisation. This may happen 
                    if the problem is ill-posed, e.g. if no common components can be found or the common directions of variance have already been identified.
        gamma_init: False or array of length n_regions. If not False, the optimization algorithm uses the array as initial condition for gamma. In the optimisation algorithm, beta is determined from the current gamma, 
                    without utilising an initial beta matrix. Default value False.
        rand_init: Boolean. If True, the decomposition will use random initial gamma values. Default value True.
        n_init: Integer. Default value 20. 
        max_iter: Integer>0. Maximal number of iterations when determining gamma (and beta, if betaLinReg is False). Default value 1000.
        tol: Float >0. Maximal tolerance when optimizing for gamma (and beta). If an iteration yields an absolute change smaller than this value in all elements of gamma and beta, the optimisation stops. Default 1e-4.
        trace_sol: Boolean. Whether or not to keep all intermediate steps in the optimization of gamma and beta. The steps are stored in lists in instance variables self.gamma_steps_all and self.beta_steps_all.
        seed_initial: Integer or None. If integer, this seeds the random initial conditions. Default value False.
        betaLinReg: Boolean. If true, the algorithm concludes with performing ordinary linear regression on the transformed values using the gamma transformation found to improve accuracy of beta estimation. Default False.
        overwrite_fit: Boolean. If False: Returns an exception if the class object has already been fitted to data. If True: Fits using the provided data and overwrites any existing values of gamma, beat, dfd_values_training, and u_training.

        Returns:
        --------
        self.beta: Array-like of shape (q,n_dir). Coefficients of the log-linear model identified during decomposition.
        self.gamma: Array-like of shape (p,n_dir). Matrix with each column being an identified gamma projection.
        self.dfd_values_training: Array of length n_dir. Contains the average values of "deviation from diagonality" computed on the data used to fit the model. This can be used for selecting the number of projections (see Zhao, Y. et al. (2021)). 
        self.u_training: The transformed values of the data used to fit the model, i.e. the logarithm of the diagonal elements of Gamma.T @ Sigma_i @Gamma for subject i.
        self.residual_std_train: Projection-wise standard deviation of the residuals (i.e. transformed values minus the mean). Computed assuming homogeneity of variance.
        self.beta_CI95_parametric: Nan or list of length 2 containing matrices whose elements are the lower and upper bounds of the elementwise confidence intervals for the beta matrix.
                        Only non-Nan if fitted with betaLinReg set to true. The limits are determined from the SE of beta coefficients identified with linear regression.
        self.beta_pvals: Nan or array-like of shape (q,n_dir). If non-Nan, the array contains the coefficient-wise p-values of the hypothesis test for the beta coefficient being equal to 0. Significance level is 0.05. 
                        Only non-Nan if fitted with betaLinReg set to true. The p-values are determined from coefficient-wise t-tests of beta coefficients identified with linear regression.
        self.beta_tvals: Nan or array-like of shape (q,n_dir). If non-Nan, the array contains the coefficient-wise t-values of the hypothesis test for the beta coefficient being equal to 0.
                        Only non-Nan if fitted with betaLinReg set to true. The t-values are determined from the SE of beta coefficients identified with linear regression.
        self.decomp_settings: Dictionary. When running the decomposition method, settings are stored in this dictionary (e.g. number of components, initial conditions, number of iterations, tolerance, etc.) 
                        
        Raises:
        -------
        Exception: Raises exception if the model has already been fitted and overwrite_fit is False.
        Exception: Raises exception if no common components (gammas) can be identified.
        Exception: Raises and handles exception, if a singular matrix occurs during the optimisation. This may happen if the problem is ill-posed, e.g. if no common components can be found or the common directions of 
                    variance have already been identified. Upon this exception, the gamma and beta already identified are kept.
        """

        self.decompose(Y_dat, X_dat, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed_initial = seed_initial, betaLinReg = betaLinReg, overwrite_fit = overwrite_fit, *kwargs)

    def transform_timeseries(self, Y_dat):
        """Takes a list of time series data and transforms it with the gamma matrix from self.

        Parameters:
        ----------- 
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.

        Returns:
        --------
        u_vals: np.array of size (n_subj, n_dirs): The values obtained by, for each projection direction j, 
                   using the projection u_i = log(gamma_j.T@Sigma_i@gamma_j)
                   for each subject i.
        """


        if self.gamma is False:
            raise Exception('Could not transform data, because the gamma matrix is not defined. Please train the model or set the gamma_matrix manually.')

        cov_matrices = fca.calc_covmatrix_listtolist(Y_dat)

        u_vals = np.log(np.array([np.diag(self.gamma.T@cov_matrices[i]@self.gamma) for i in range(len(Y_dat))]))

        return u_vals

    def transform_FC(self, corr_list):
        """Takes a list of covariance/correlation matrices and transforms it with the gamma matrix from self.

        Parameters:
        ----------- 
        corr_list: List of len n_subj containing covariance/correlation matrices, each of size (p, p). Elements 
        should be Pearson full correlation or covariance matrices with n degrees of freedom (population covariance matrices). 

        Returns:
        --------
        u_vals: np.array of size (n_subj, n_dirs): The values obtained by, for each projection direction j, 
                   using the projection u_i = log(gamma_j.T@Sigma_i@gamma_j)
                   for each subject i.
        """

        if self.gamma is False:
            raise Exception('Could not transform data, because the gamma matrix is not defined. Please train the model or set the gamma_matrix manually.')

        u_vals = np.log(np.array([np.diag(self.gamma.T@corr_list[i]@self.gamma) for i in range(len(corr_list))]))

        return u_vals
    
    def predict(self, X_dat):
        """Takes the feature matrix, X_dat, and predicts u values from the fitted model (i.e. using the coefficients in self.beta)

        Parameters:
        ----------- 
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (intercept).

        Returns:
        --------
        u_pred: Predicted u values.
        """

        u_pred = X_dat@self.beta

        return u_pred

    def calc_Zscores(self, X_dat, u_vals):
        """Takes transformed data (after projections) and calculates Z-scores based on model prediction and standard deviations from the training data.
           Variance is assumed to be homogenous within each projection.

        Parameters:
        ----------- 
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (does not work without the intercept).
        u_vals: np.array of size (n_subj, n_dirs): The values obtained by, for each projection direction j, 
            using the projection u_i = log(gamma_j.T@Sigma_i@gamma_j)
            for each subject i.

        Returns:
        --------
        Zscores: np.array of size (n_subj, n_dirs): The Z-scores obtained by subtracting the expected value given features in X and dividing by the (projection-specific) standard deviation from the training data.
        
        Raises:
        -------
        Exception: Raises exception if the model has not been fitted (because estimation of beta and standard deviations in training data is required).
        """    

        isfit = self.isfitted()
        if not isfit:
            raise Exception('Could not calculate Z-scores, because the model has not been fitted. Please train the model before calculating Z-scores.')
        
        u_pred = self.predict(X_dat)
        Z_scores = np.array([(u_vals[i,:] - u_pred[i,:])/self.residual_std_train for i in range(u_vals.shape[0])])
        print('WARNING: Calculated Z-scores based on the provided data by using the standard deviation from the training data. This is based on the assumption of homogenous variance of the residuals of the transformed training data.')

        return Z_scores

    def decompose_bootstrap(self, Y_dat, X_dat, n_samples, max_comps, CI_lvl = 0.05, gamma_init = False, rand_init = True, n_init = 20, max_iter=1000, tol = 1e-4, trace_sol = 0, seed_initial = None, betaLinReg = True, seed_bootstrap = None, overwrite_fit=False):
        """Performs FUNCOIN decomposition and bootstrapping of beta coefficients given covariate matrix, X_dat, and a list of time series data, Y_dat.
        To account for the case where the bootstrap sampling changes the order of the gammas identified, the gamma vectors of 
        the bootstrapped gammas are sorted consecutively to maximize the dot product with the gammas identified on the original dataset.
        For already fitted or predefined gamma, bootstrapping of beta coefficients can be performed by calling .bootstrap_only([args])
        
        Parameters:
        -----------
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (does not work without the intercept).
        n_samples: Integer >0. Number of bootstrap samples.
        max_comps: Maximal number of components (gamma), to be identified. May terminate with fewer components if a singular matrix occurs during the optimisation. This may happen 
                    if the problem is ill-posed, e.g. if no common components can be found or the common directions of variance have already been identified.
        CI_lvl: The significance level used for the end points of the confidence interval. Default value 0.05.
        gamma_init: False or array of length n_regions. If not False, the optimization algorithm uses the array as initial condition for gamma. In the optimisation algorithm, beta is determined from the current gamma, 
                    without utilising an initial beta matrix. Default value False.
        rand_init: Boolean. If True, the decomposition will use random initial gamma values. Default value True.
        n_init: Integer. Default value 20. 
        max_iter: Integer>0. Maximal number of iterations when determining gamma (and beta, if betaLinReg is False). Default value 1000.
        tol: Float >0. Maximal tolerance when optimizing for gamma (and beta). If an iteration yields an absolute change smaller than this value in all elements of gamma and beta, the optimisation stops. Default 1e-4.
        trace_sol: Boolean. Whether or not to keep all intermediate steps in the optimization of gamma and beta. 
        seed_initial: Integer or None. If integer, this seeds the random initial conditions. Default value False.
        betaLinReg: Boolean. If true, the algorithm concludes with performing ordinary linear regression on the transformed values using the gamma transformation found to improve accuracy of beta estimation. Default False.
        seed_bootstrap: Integer or None. If integer, this seeds the bootstrap sampling algorithm, thereby derermining the random bootstrap samples drawn.
        
        Returns:
        --------
        self.beta: Array-like of shape (q,n_dir). Coefficients of the log-linear model identified during decomposition.
        self.gamma: Array-like of shape (p,n_dir). Matrix with each column being an identified gamma projection.
        self.betas_bootstrap: List of length n_samples. Contains all beta matrices determined with bootstrapping.
        self.beta_CI_bootstrap = List of length 2 containing matrices of size (q,n_dir), i.e. same size as self.beta. The elements of the two matrices in the list are the lower and upper bound of the confidence interval of the gamma matrix determined by bootstrapping.
        self.CI_lvl = Float. Must be between 0 and 1. The significance level used for the end points of the confidence interval. If not specified when calling the decompose_bootstrap method, the default value is 0.05.
        
        Raises:
        -------
        Exception: Raises exception if the model has already been fitted and overwrite_fit is False.
        Warning: Raises a warning, if a singular matrix occurs during the optimisation. This may happen if: 1) the problem is ill-posed, e.g. if no common components can be found, in which case self.beta and self.gamma are assigned NaN values, or 2) the common directions of 
                    variance have already been identified, in which case the gamma and beta already identified are kept.
        """
        
        self.decompose(Y_dat, X_dat, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed_initial = seed_initial, betaLinReg = betaLinReg, overwrite_fit=overwrite_fit)

        self.bootstrap_only(Y_dat, X_dat, n_samples, CI_lvl = CI_lvl, max_iter=max_iter, tol = tol, betaLinReg = betaLinReg, seed_bootstrap = seed_bootstrap)


    def bootstrap_only(self, Y_dat, X_dat, n_samples, CI_lvl = 0.05, max_iter=1000, tol = 1e-4, betaLinReg = True, seed_bootstrap = None, bias_corrections = True):
        """Performs bootstrapping of beta coefficients given covariate matrix, X_dat, a list of time series data, Y_dat, and predefined or fitted gamma and beta matrices (stored in the FUNCOIN instance self.gamma, self.beta) .
        
        Parameters:
        -----------
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (does not work without the intercept).
        n_samples: Integer >0. Number of bootstrap samples.
        max_comps: Maximal number of components (gamma), to be identified. May terminate with fewer components if a singular matrix occurs during the optimisation. This may happen 
                    if the problem is ill-posed, e.g. if no common components can be found or the common directions of variance have already been identified.
        CI_lvl: The significance level used for the end points of the confidence interval. Default value 0.05.
        max_iter: Integer>0. Maximal number of iterations when determining gamma (and beta, if betaLinReg is False). Default value 1000.
        tol: Float >0. Maximal tolerance when optimizing for gamma (and beta). If an iteration yields an absolute change smaller than this value in all elements of gamma and beta, the optimisation stops. Default 1e-4.
        betaLinReg: Boolean. If true, the algorithm concludes with performing ordinary linear regression on the transformed values using the gamma transformation found to improve accuracy of beta estimation. Default False.
        seed_bootstrap: Integer or None. If integer, this seeds the bootstrap sampling algorithm, thereby derermining the random bootstrap samples drawn.
        
        Returns:
        --------
        self.beta: Array-like of shape (q,n_dir). Coefficients of the log-linear model identified during decomposition.
        self.gamma: Array-like of shape (p,n_dir). Matrix with each column being an identified gamma projection.
        self.betas_bootstrap: List of length n_samples. Contains all beta matrices determined with bootstrapping.
        self.beta_CI_bootstrap: List of length 2 containing matrices of size (q,n_dir), i.e. same size as self.beta. The elements of the two matrices in the list are the lower and upper bounds of the confidence interval of the beta matrix determined by bootstrapping.
        self.CI_lvl: Float. Must be between 0 and 1. The significance level, i.e. resulting in a 1-CI_lvl confidence interval. Default value is 0.05.
        
        Raises:
        -------
        Exception: Raises exception if gamma is not defined by either training the model or by manually defining it.
        Warning: Raises a warning, if a singular matrix occurs during the optimisation. This may happen if: 1) the problem is ill-posed, e.g. if no common components can be found, in which case self.beta and self.gamma are assigned NaN values, or 2) the common directions of 
                    variance have already been identified, in which case the gamma and beta already identified are kept.
        Warning: Raises a warning if a singular matrix occurs during the 
        """

        if (self.gamma is False) or (self.beta is False):
            raise Exception('Could not run bootstrapping procedure, because the gamma and/or beta matrices are not defined. Before bootstrapping, please train the model or set the gamma and beta manually. Decomposition followed by bootstrapping can be called with the method .decompose_bootstrap([args]).')

        max_comps_bootstrap = self.gamma.shape[1]

        Ti_vec = [Y_dat[i].shape[0] for i in range(len(Y_dat))]

        n_subj = len(Y_dat)

        beta_mats_bootstrap = []

        rng = np.random.default_rng(seed = seed_bootstrap)
        
        for i2 in range(n_samples):
            beta_bootstrap = np.zeros((X_dat.shape[1],self.gamma.shape[1]))*np.nan
            sample_inds = rng.choice(n_subj, n_subj)
            Y_sample = [Y_dat[i] for i in sample_inds]
            X_sample = X_dat[sample_inds,:]
            Ti_list = [Ti_vec[i] for i in sample_inds]
            

            for i3 in range(max_comps_bootstrap):
                
                if not bias_corrections:
                    if i3 == 0:
                        Si_list_sample = fca.make_Si_list(Y_sample)
                    else:
                        Si_list_sample = self.__make_Si_list_tilde(Y_sample, self.gamma[:,:i3], self.beta[:,:i3])
                else:
                    Si_list_sample = fca.make_Si_list(Y_sample)


                try:
                    if betaLinReg:
                        _, beta_new = self.__update_beta_LinReg(Si_list_sample, X_sample, Ti_list, gamma_init=self.gamma[:,i3])
                    else:
                        beta_old = np.expand_dims(self.beta[:,i3],1)
                        gamma_old = np.expand_dims(self.gamma[:,i3],1)
                        _, beta_new = self.__optimize_only_beta(Si_list_sample, X_sample, Ti_list, beta_init=beta_old, gamma_init=gamma_old, max_iter=max_iter, tol=tol)
                except:
                    beta_new = np.zeros(X_dat.shape[1])*np.nan


                beta_bootstrap[:,i3] = beta_new.flatten()

            beta_mats_bootstrap.append(beta_bootstrap)

        num_nan = np.sum(np.isnan(beta_bootstrap[0,:]))
        if num_nan > 0:
            warnings.warn(f'{num_nan} bootstrap samples returned NaN values. Confidence intervals are based on {n_samples-num_nan} bootstrap samples')

        valsbeta = np.array([beta_mats_bootstrap[i] for i in range(len(beta_mats_bootstrap))])
        beta_mat_lowCI = np.nanpercentile(valsbeta, (100*CI_lvl)/2, axis=0)
        beta_mat_highCI = np.nanpercentile(valsbeta, (100*(1-CI_lvl/2)), axis=0)

        beta_mat_CI = [beta_mat_lowCI, beta_mat_highCI]


        self.betas_bootstrap = beta_mats_bootstrap
        self.beta_CI_bootstrap = beta_mat_CI
        self.CI_lvl_bootstrap = CI_lvl

    def add_projections(self, n_add, Y_dat, X_dat):
        """Identifies projection direction in addition to the projections already found by fitting the FUNCOIN instance.

        Parameters:
        -----------
        n_add: Integer >0. Integer specifying the maximal number of more direction to be identified from the provided training data.
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (does not work without the intercept).
       
        Returns:
        --------
        self.gamma: The old self.gamma is overwritten by a new matrix where the newly identified gamma projections are added.
        self.beta: The old self.beta is overwritten by a new matrix where the coefficients for the newly identified projections are added.

        Raises:
        -------
        Exception: If the model is not fitted before calling .add_projection.
        Exception: If the data provided for adding directions does not match the data used to train the model in the first place.
        
        """

        gamma_old = self.gamma
        isfit = self.isfitted()
        if not isfit:
            raise Exception('Could not add projections, because the FUNCOIN model has not been fitted. Please run the decompose method instead.')

        u_test = self.transform_timeseries(Y_dat)

        test = np.all(self.u_training==u_test)

        if not test:
            raise Exception('Did not add any projection directions. The data provided do not match the data used to fit the model in the first place.')

        gamma_init = self.decomp_settings['gamma_init']
        rand_init = self.decomp_settings['rand_init']
        n_init = self.decomp_settings['n_init']
        max_iter = self.decomp_settings['max_iter']
        tol = self.decomp_settings['tol']
        trace_sol = self.decomp_settings['trace_sol']
        seed_initial = self.decomp_settings['seed_initial']
        betaLinReg = self.decomp_settings['betaLinReg']

        max_comps_new = gamma_old.shape[1] + n_add

        self.decompose(self, Y_dat, X_dat, max_comps=max_comps_new, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed_initial = seed_initial, betaLinReg = betaLinReg, overwrite_fit=False, add_to_fit = True)

    def calc_dfd_values(self, Y_dat, weighted_io=1, dfd_aritm = 0, logtrick_io = 1):
        """
        Computes the  "deviation from diagonality (dfd)" (Flury and Gautschi, 1986) averaged across subjects for each of the identified directions in the FUNCOIN model. This is suggested as a measure to decide on the number of projection directions to keep (Zhao et al, 2021).
        Can also be called as calc_dfd_values_ts (to distinguish from calc_dfd_values_FC).

        Parameters:
        -----------
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.List of length n_subects consisting of arrays of shape (n_timepoints x n_parcels) with the data used in the fitting process
        weighted_io: Boolean or 0/1. If set to True/1, the average dfd value is a weighted average according to the number of time points for each subject. If all subjects have the same number of time points, setting this parameter to 1 can improve estimation by reducing the risk of overflow.
        dfd_aritm: Boolean or 0/1. If set to False/0, the average dfd value is computed as the geometric mean (weighted or unweighted according to the weighted_io parameter). If 1, the arithmetic mean is used. Default value is 0. 
        logtrick_io: Boolean of 0/1. When computing the harmonic mean, a log-transformation is temporarily applied to avoid overflow. Recommended, but can be disabled to test the difference. Default value 1.
    
        Returns: 
        --------
        dfd_proj: A number to measure the deviation from diagonality of the projected data matrices. 
        
        Raises:
        -------
        Exception: If gamma matrix is not fitted nor predefined.
        """


        if self.gamma is False:
            raise Exception('DfD values could not be computed, because the gamma matrix is not defined. Please train the model or set the gamma_matrix manually.')

   

        n_dir = self.gamma.shape[1]
        dfd_vals = [self.__deviation_from_diag(i, Y_dat, weighted_io, dfd_aritm, logtrick_io) for i in range(n_dir)]
        return dfd_vals
    
    def calc_dfd_values_ts(self, Y_dat, weighted_io=1, dfd_aritm = 0, logtrick_io = 1):
        """
        Computes the  "deviation from diagonality (dfd)" (Flury and Gautschi, 1986) averaged across subjects for each of the identified directions in the FUNCOIN model. This is suggested as a measure to decide on the number of projection directions to keep (Zhao et al, 2021).
        
        Parameters:
        -----------
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.List of length n_subects consisting of arrays of shape (n_timepoints x n_parcels) with the data used in the fitting process
        weighted_io: Boolean or 0/1. If set to True/1, the average dfd value is a weighted average according to the number of time points for each subject. If all subjects have the same number of time points, setting this parameter to 1 can improve estimation by reducing the risk of overflow.
        dfd_aritm: Boolean or 0/1. If set to False/0, the average dfd value is computed as the geometric mean (weighted or unweighted according to the weighted_io parameter). If 1, the arithmetic mean is used. Default value is 0. 
        logtrick_io: Boolean of 0/1. When computing the harmonic mean, a log-transformation is temporarily applied to avoid overflow. Recommended, but can be disabled to test the difference. Default value 1.
    
        Returns: 
        --------
        dfd_proj: A number to measure the deviation from diagonality of the projected data matrices. 
        
        Raises:
        -------
        Exception: If gamma matrix is not fitted nor predefined.
        """


        dfd_vals = self.calc_dfd_values(self, Y_dat, weighted_io, dfd_aritm, logtrick_io)
        return dfd_vals

    def calc_dfd_values_FC(self, FC_list, weighted_io=1, dfd_aritm = 0, logtrick_io = 1, Ti_list=[]):
        """
        Computes the  "deviation from diagonality (dfd)" (Flury and Gautschi, 1986) averaged across subjects for each of the identified directions in the FUNCOIN model. This is suggested as a measure to decide on the number of projection directions to keep (Zhao et al, 2021).
        
        Parameters:
        -----------
        FC_list: List of length [number of subjects] containing covariance/correlation matrix for each subject. Each element of the list should be array-like of shape (p, p), with p the number of regions.
        weighted_io: Boolean or 0/1. If set to True/1, the average dfd value is a weighted average according to the number of time points for each subject. If all subjects have the same number of time points, setting this parameter to 1 can improve estimation by reducing the risk of overflow.
        dfd_aritm: Boolean or 0/1. If set to False/0, the average dfd value is computed as the geometric mean (weighted or unweighted according to the weighted_io parameter). If 1, the arithmetic mean is used. Default value is 0. 
        logtrick_io: Boolean of 0/1. When computing the harmonic mean, a log-transformation is temporarily applied to avoid overflow. Recommended, but can be disabled to test the difference. Default value 1.
    
        Returns: 
        --------
        dfd_proj: A number to measure the deviation from diagonality of the projected data matrices. 
        
        Raises:
        -------
        Exception: If gamma matrix is not fitted nor predefined.
        """


        if self.gamma is False:
            raise Exception('DfD values could not be computed, because the gamma matrix is not defined. Please train the model or set the gamma_matrix manually.')

   

        n_dir = self.gamma.shape[1]
        dfd_vals = [self.__deviation_from_diag(i, FC_list, weighted_io, dfd_aritm, logtrick_io, FC_mode = True, Ti_list=Ti_list) for i in range(n_dir)]
        return dfd_vals


    def score(self, X_dat, u_true = None, Y_dat = None, score_type = 'r2_score', **kwargs):
        """ Calculate score functions for each gamma projection to asses goodness of fit of the log-linear model.
        
        Parameters:
        -----------
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (does not work without the intercept).
        u_true: Array-like of shape (n_subjects, n_comps). 
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.
        score_type: String specifying a type of scoring, e.g. 'r2_score' or 'mean_absolute_error. Any regression metric method from the sklearn.metrics submodule can be used.
                    Full list of possible arguments as of sklearn v1.4: ['explained_variance_score', 'max_error', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 
                                                     'mean_squared_log_error', 'root_mean_squared_log_error', 'median_absolute_error', 'r2_score', 'mean_poisson_deviance', 
                                                     'mean_gamma_deviance', 'mean_absolute_percentage_error', 'd2_absolute_error_score']
                    For specifics on each score method, see sklearn documentation (https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
                    Default score type is R^2
        **kwargs: Keyword arguments, e.g. sample_weight or force_finite (for r2_score). These are used when calling the actual score function. For each choice of score type, the arguments 
                  listed in the sklearn documentation can be used.
                  Note that keyword argument 'multioutput' will be ignored, since this function outputs scores for each projection.
        
        Returns:
        --------
        scores: Array of length n_comps. The value of the chosen metric for each gamma projection when comparing the provided true values to the prediction based on the model and the X_dat matrix. 
        """
        
        module = importlib.import_module('sklearn.metrics')
        
        try:
            scorefunc = getattr(module, score_type)
        except:
            raise Exception('Scoring type not found in sklearn.metrics.')

        if (self.gamma is False) or (self.beta is False):
            raise Exception('Could not calculate score, because the Gamma and/or Beta matrix is not defined. Please fit the model or define these manually.')

        if (u_true is None) and (Y_dat is None):
            raise Exception('Could not calculate score, no true data values were given. Please provide transformed data (u values) or Y_dat (list of time series data).')

        if (u_true is not None) and (Y_dat is not None):
            warnings.warn('Warning. Both u values and time series data were provided. Calculating score based on the provided u values.')

        if 'multioutput' in kwargs.keys():
            del kwargs['multioutput']

        if score_type == 'r2_score':
            'Providing r2_score, which is the default of this function.'

        u_pred = self.predict(X_dat)

        if u_true is None:
            u_true = self.transform_timeseries(Y_dat)


        scores = scorefunc(u_true, u_pred, multioutput='raw_values', **kwargs)

        return scores

    def CI_beta_parametric(self, X_dat, CI_lvl=0.05):
        """
        Computes the parametric confidence intervals of the beta coefficients. Relies on the assumption of homogenous and normally distributed residuals. 
        This is automatically called with fitting the model (i.e. running decomposition) with betaLinReg set to true, and the 95%-CI is stored in self.beta_CI95_parametric.

        Parameters:
        -----------
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (intercept).
        CI_lvl: Float. Must be between 0 and 1. The significance level, i.e. resulting in a 1-CI_lvl confidence interval. Default value is 0.05.

        Returns: 
        --------
        beta_CI_parametric: List of length 2 containing matrices of size (q,n_dir), i.e. same size as self.beta. The elements of the two matrices in the list are the lower and upper bounds of the confidence interval of the beta determined parametrically.

        Raises:
        -------
        Exception: If the model has not been fitted.
        Exception: If the model has been fitted without the betaLinReg set to True. 
        """

        fit_stat = self.isfitted()
        if not fit_stat:
            raise Exception('Cannot calculate confidence interval, because the model has not been fitted. Run the decomposition method first.')
        else:
            if not self.decomp_settings['betaLinReg']:
                raise Exception('Cannot compute parametric confidence intervals. Only valid when model is fitted with betaLinReg set to True.')

        s2 = self.__sample_s2_coefficients(X_dat=X_dat)
        Q_inv = self.__calc_Qinv(X_dat)

        CI_low_s2 = np.zeros_like(self.beta)
        CI_high_s2 = np.zeros_like(self.beta)

        df = X_dat.shape[0]-X_dat.shape[1]

        for i in range(CI_low_s2.shape[1]): 
            SE_beta = np.sqrt(s2[i]*np.diag(Q_inv))
            CI_lim = 1-CI_lvl/2
            CI_low_s2[:,i] = self.beta[:,i] - t.ppf(CI_lim, df) * SE_beta
            CI_high_s2[:,i] = self.beta[:,i] + t.ppf(CI_lim, df)  * SE_beta

        beta_CI_parametric = [CI_low_s2, CI_high_s2]

        return beta_CI_parametric

    def ttest_beta(self, X_dat, h0_beta = False):
        """
        Performs two-tailed t-test of the beta coefficients. Relies on the same assumptions as linear regression and parametric confidence intervals. 
        This is automatically called with fitting the model (i.e. running decomposition) with betaLinReg set to true and the p-values are stored in self.beta_pvals.

        Parameters:
        -----------
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (intercept).
        h0_beta: False or array-like of shape (q,n_dir), i.e. same size as self.beta. Contains the hypothesized values to test for, i.e. testing if the fitted beta is significantly different from the value in h0_beta. Default is False, in which case the beta=0 hypotheses are tested for all elements of fitted beta.   

        Returns: 
        --------
        beta_pvals: Array-like of shape (q,n_dir). Contains the p-values of each beta coefficient from the fit. 
        beta_tvals: Array-like of shape (q,n_dir). Contains the t-values of the t-test of each beta coefficient from the fit. 
        
        Raises:
        -------
        Exception: If the model has not been fitted.
        Exception: If the model has been fitted without the betaLinReg set to True. 
        """

        fit_stat = self.isfitted()
        if not fit_stat:
            raise Exception('Cannot parametrically determine p-values, because the model has not been fitted. Run the decomposition method first.')
        else:
            if not self.decomp_settings['betaLinReg']:
                raise Exception('Cannot parametrically determine parametric confidence intervals. Only valid when model is fitted with betaLinReg set to True.')


        if h0_beta is False:
            h0_beta = np.zeros_like(self.beta)

        s2 = self.__sample_s2_coefficients(X_dat=X_dat)
        Q_inv = self.__calc_Qinv(X_dat)
        

        beta_tvals = np.zeros_like(self.beta)
        for i in range(beta_tvals.shape[1]): 
            SE_beta = np.sqrt(s2[i]*np.diag(Q_inv))
            beta_tvals[:,i] = (self.beta[:,i]-h0_beta[:,i])/SE_beta

        df = X_dat.shape[0]-X_dat.shape[1]

        beta_pvals = (1 - t.cdf(np.abs(beta_tvals), df=df))*2

        return beta_pvals, beta_tvals


    def simulate_data(self, X_dat, n_T, seed = None):
        """ Simulate data based on Gamma and Beta matrices(predefined or fitted) and X matrix (feature matrix).
        
        Parameters:
        -----------
        X_dat: Array-like of shape (n_subjects, n_covariates+1). First column has to be ones (does not work without the intercept).
        n_T: Integer or list/array of integers of length equal to the number of rows in X_dat. The number of time points to be simulated for each "subject" (corresponding to features in each row in  X_dat).
             If an integer is provided, all "subjects" will have n_T time points.
        seed: None, int or array_like[ints]. If different from None, seeds the simulation of time series data. Default value None. 
              Also accepts SeedSequence instances and BitGenerators. See documentation for numpy.default_rng() for dertails. 

        Returns:
        --------
        Y_sim: List of length [number of subjects] containing time series data for each "subject". Each element of the list is array-like of shape (n_T[i], q).
        """""

        if (self.gamma is False) or (self.beta is False):
            raise Exception('Could not simulate data. Both Gamma and Beta matrices must be defined by manually definition or fitting.')

        if X_dat.shape[1] != self.beta.shape[0]:
            raise Exception(f'Could not simulate data. Number of features ({X_dat.shape[1]}) must match the first dimension of the beta matrix ({self.beta.shape[0]}.')

        if self.gamma.shape[0]!=self.beta.shape[1]:
            raise Exception(f'Could not simulate data. Length of first dimension of gamma ({self.gamma.shape[0]}) must match the length of the second dimension of the beta matrix ({self.beta.shape[1]}.')

        if self.isfitted():
            word = 'fitted'
        else:
            word = 'predefined'

        print(f'Simulating data using {word} Gamma and Beta matrices.')

        n_subj = X_dat.shape[0]
        p_model = self.gamma.shape[0]

        lambdas_subj = np.array([np.exp(X_dat[i,:]@self.beta) for i in range(n_subj)])

        Sigma_list = [self.gamma@np.diag(lambdas_subj[i])@self.gamma.T for i in range(n_subj)]   

        mean_sim = np.zeros(p_model)

        rng = np.random.default_rng(seed = seed)

        if type(n_T) == int:
            n_T = [n_T]*len(Sigma_list)

        Y_sim = [rng.multivariate_normal(mean_sim, Sigma_list[i], n_T[i]) for i in range(n_subj)]

        return Y_sim

    def isfitted(self):
        """
        When called, checks if the model has been fitted and returns True of False
        """
        return self.__fitted

    #Private methods

    def __decomposition(self, Y_dat, X_dat, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, trace_sol = 0, seed = None, betaLinReg = True, overwrite_fit = False, add_to_fit = False, FC_mode = False, Ti_list=[], ddof = 0):

        if (not overwrite_fit) and (add_to_fit):
            gamma_mat = self.gamma
            beta_mat = self.beta
        else:
            gamma_mat = False

        if np.ndim(gamma_mat)>0:
            n_dir_init = gamma_mat.shape[1]
            gamma_mat_new = gamma_mat
        else:
            n_dir_init = 0

        best_llh_directions = []
        best_beta_steps_all = []
        best_gamma_steps_all = []

        if FC_mode == False:
            Y_dat = [Y_dat[i]-np.mean(Y_dat[i],0) for i in range(len(Y_dat))]
            Ti_list = [Y_dat[i].shape[0] for i in range(len(Y_dat))]
            Si_list = fca.make_Si_list(Y_dat)
        else:
            Si_list = fca.make_Si_list_from_FC_list(Y_dat, Ti_list, ddof)

        for i in range(n_dir_init,max_comps):

            if i == 0:
                try:
                    _, best_beta, best_gamma, _, _, _, _, _, _, best_beta_steps, best_gamma_steps = self.__first_direction(Si_list, X_dat, Ti_list, gamma_init, rand_init, n_init, max_iter = max_iter, tol = tol, trace_sol=trace_sol, seed=seed, betaLinReg=betaLinReg)
                except:
                    raise Exception('Exception occured. Did not find any principal directions using FUNCOIN algorithm.')
                else:
                    beta_mat_new = best_beta
                    gamma_mat_new = best_gamma
            else:

                if seed:
                    seed += 1 #Ensure new random initial conditions for each projection identified. If seeded to begin with, the decomposition as a whole is still reproducable.
                try:
                    beta_mat_new, gamma_mat_new, _, _, _ = self.__kth_direction(Y_dat, X_dat, beta_mat, gamma_mat, gamma_init, rand_init, n_init = n_init, max_iter=max_iter, tol = tol, trace_sol=trace_sol, seed=seed, betaLinReg=betaLinReg, FC_mode = FC_mode, Ti_list=Ti_list, ddof = ddof)
                except:
                    beta_mat = beta_mat_new
                    gamma_mat = gamma_mat_new
                    # best_llh_directions.append(best_llh)
                    # best_beta_steps_all.append(best_beta_steps)
                    # best_gamma_steps_all.append(best_gamma_steps)
                    warnings.warn(f'Identified {gamma_mat.shape[1]} components ({max_comps} were requested).')
                    return gamma_mat, beta_mat

            beta_mat = beta_mat_new
            gamma_mat = gamma_mat_new
            # best_llh_directions.append(best_llh)
            # best_beta_steps_all.append(best_beta_steps)
            # best_gamma_steps_all.append(best_gamma_steps)

        return gamma_mat, beta_mat


    def __first_direction(self, Si_list, X_dat, Ti_list, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol = 1e-4, trace_sol = False, seed = None, betaLinReg = False):        
        """                     
        Using the method from Zhao et al 2021 to find the first gamma projection.
        """

        p_model = Si_list[0].shape[0]
        q_model = X_dat.shape[1]

        beta_init = np.zeros([q_model,1])

        if type(gamma_init) == bool and not rand_init:
            gamma_init = np.ones([p_model,1])
        if rand_init:
            gamma_init = self.__random_initial_conds(p_model, n_init, seed=seed)

        Xi_list = fca.make_Xi_list(X_dat)
        Ti_list_arr = np.array(Ti_list)
        Si_list_arr = np.array(Si_list)
        sigma_bar = np.sum(Si_list_arr, axis=0)/np.sum(Ti_list_arr)

        H_mat = sigma_bar

        eigvalsH, eigvecsH = np.linalg.eigh(H_mat)
        eigvals_new = eigvalsH
        eigvals_new = 1/(np.sqrt(eigvalsH))


        H_pow = eigvecsH@np.diag(eigvals_new)@eigvecsH.T

        best_gamma_all = []
        best_beta_all = []
        best_llh_all = []
        if trace_sol:
            beta_steps_all = []
            gamma_steps_all = []
        llh_steps_all = []
        llh_steps_split_all = []
        llh_steps_beta_optim_all = []

        n_init = min(n_init, gamma_init.shape[1])

        for l in range(n_init):

            beta_old = beta_init
            gamma_old = np.expand_dims(gamma_init[:,l],1)
            gamma_norm = np.linalg.norm(gamma_old)
            if gamma_norm != 0:
                gamma_old = gamma_old / gamma_norm

            gamma_old = H_pow@gamma_old

            llh_steps = [np.squeeze(self.__loglikelihood(beta_old, gamma_old, X_dat, Ti_list, Si_list))]
            llh_steps_split = [np.squeeze(self.__loglikelihood(beta_old, gamma_old, X_dat, Ti_list, Si_list))]
            beta_steps = [beta_init]
            gamma_steps = [np.expand_dims(gamma_init[:,l],1)]

            step_ind = 0
            diff = 100

            while step_ind<max_iter and diff > tol:

                #Update beta

                try:

                    matlist_arr = np.array([(np.exp(-Xi_list[i].T @ beta_old) * gamma_old.T@ Si_list[i] @gamma_old) * Xi_list[i] @ Xi_list[i].T  for i in range(X_dat.shape[0])])
                    mat_for_inv = np.sum(matlist_arr, axis=0)
                    part1 = np.linalg.inv(mat_for_inv)

                except:
                    raise Exception('Singular matrix occured.')

                matlist2_arr = np.array([(Ti_list[i] - np.exp(-Xi_list[i].T @ beta_old)@gamma_old.T@ Si_list[i] @gamma_old) * Xi_list[i] for i in range(X_dat.shape[0])])
                part2 = np.sum(matlist2_arr, axis=0)

                beta_new = beta_old - part1@part2


                llh_steps_split.append(np.squeeze(self.__loglikelihood(beta_new, gamma_old, X_dat, Ti_list, Si_list)))

                #Update gamma
                A_matlist_arr = np.array([np.exp(-Xi_list[i].T @ beta_new) * Si_list[i] for i in range(X_dat.shape[0])])
                A_mat = np.sum(A_matlist_arr, axis=0)
                HAH_mat = H_pow @ A_mat @ H_pow

                eigvals, eigvecs = np.linalg.eigh(HAH_mat)
                best_ind = np.argmin(eigvals)

                gamma_new = np.expand_dims(H_pow @ eigvecs[:,best_ind],1)
                llh_steps_split.append(np.squeeze(self.__loglikelihood(beta_new, gamma_new, X_dat, Ti_list, Si_list)))
                llh_steps.append(np.squeeze(self.__loglikelihood(beta_new, gamma_new, X_dat, Ti_list, Si_list)))

                beta_steps.append(beta_new)
                gamma_steps.append(gamma_new)

                gamma_diff = np.max(np.squeeze(abs(gamma_old-gamma_new)))
                beta_diff = np.max(np.squeeze(abs(beta_old-beta_new)))
                diff = np.maximum(gamma_diff, beta_diff)
                step_ind +=1

                gamma_old = gamma_new
                beta_old = beta_new

            gamma_old = gamma_old/np.linalg.norm(gamma_old)
            if gamma_old[0] < 0:
                gamma_old = -gamma_old

            if betaLinReg:
                llh_steps_beta_optim, beta_new = self.__update_beta_LinReg(Si_list, X_dat, Ti_list, gamma_old)
            else:
                llh_steps_beta_optim, beta_new = self.__optimize_only_beta(Si_list, X_dat, Ti_list, beta_old, gamma_old)
        
            best_llh_here = llh_steps[-1]
            best_gamma_here = gamma_old
            best_beta_here = beta_new

            best_gamma_all.append(best_gamma_here)
            best_beta_all.append(best_beta_here)
            best_llh_all.append(best_llh_here)
            if trace_sol:
                beta_steps_all.append(beta_steps)
                gamma_steps_all.append(gamma_steps)
                
            llh_steps_all.append(llh_steps)
            llh_steps_split_all.append(llh_steps_split)
            llh_steps_beta_optim_all.append(llh_steps_beta_optim)

        best_llh_ind = np.argmin(best_llh_all)
        best_llh = best_llh_all[best_llh_ind]
        best_gamma = best_gamma_all[best_llh_ind]
        best_beta = best_beta_all[best_llh_ind]

        if trace_sol:
            best_gamma_steps = gamma_steps_all[best_llh_ind]
            best_beta_steps = beta_steps_all[best_llh_ind]
            gsa = self.gamma_steps_all
            bsa = self.beta_steps_all
            gsa.append(gamma_steps_all)
            bsa.append(beta_steps_all)
            self.gamma_steps_all = gsa
            self.beta_steps_all = bsa
        else:
            best_gamma_steps = float('NaN')
            best_beta_steps = float('NaN')

        return best_llh, best_beta, best_gamma, best_llh_all, best_beta_all, best_gamma_all, llh_steps_all, llh_steps_split_all, llh_steps_beta_optim_all, best_beta_steps, best_gamma_steps

    def __kth_direction(self, Y_dat, X_dat, beta_mat, gamma_mat, gamma_init = False, rand_init = True, n_init = 20, max_iter=1000, tol = 1e-4, trace_sol = 0, seed = None, betaLinReg=False, FC_mode = False, Ti_list = [], ddof = 0):
        """
        Using the method from Zhao et al 2021 to find the kth gamma projection.
        """


        if FC_mode == False:
            Ti_list = [Y_dat[i].shape[0] for i in range(len(Y_dat))]

            Si_list_tilde = self.__make_Si_list_tilde(Y_dat, gamma_mat, beta_mat)
        else:
            Si_list_tilde = self.__make_Si_list_tilde_fromFC(Y_dat, gamma_mat, beta_mat, Ti_list, ddof)

        best_llh, best_beta, best_gamma, _, _, _, _, _, _, best_beta_steps, best_gamma_steps = self.__first_direction(Si_list_tilde, X_dat, Ti_list, gamma_init, rand_init, n_init, max_iter, tol, trace_sol, seed=seed, betaLinReg=betaLinReg)
        
        gamma_mat_new = np.append(gamma_mat, best_gamma, 1)
        beta_mat_new = np.append(beta_mat, best_beta, 1)

        return beta_mat_new, gamma_mat_new, best_llh, best_beta_steps, best_gamma_steps

    def __optimize_only_beta(self, Si_list, X_dat, Ti_list, beta_init, gamma_init, max_iter = 1000, tol = 1e-4):

        Xi_list = fca.make_Xi_list(X_dat)
        llh_vals = [self.__loglikelihood(beta_init, gamma_init, X_dat, Ti_list, Si_list)]

        beta_old = beta_init
        gamma_cand = gamma_init
        step_ind = 0
        diff = 100
        while step_ind<max_iter and diff > tol:
            try:
                matlist_arr = np.array([(np.exp(-Xi_list[i].T @ beta_old) * gamma_cand.T@ Si_list[i] @gamma_cand) * Xi_list[i] @ Xi_list[i].T  for i in range(X_dat.shape[0])])
                part1 = np.linalg.inv(np.sum(matlist_arr, axis=0))
                matlist2_arr = np.array([(Ti_list[i] - np.exp(-Xi_list[i].T @ beta_old)@gamma_cand.T@ Si_list[i] @gamma_cand) * Xi_list[i] for i in range(X_dat.shape[0])])
                part2 = np.sum(matlist2_arr, axis=0)
            except:
                raise Exception('Singular matrix occured.')

            beta_new = beta_old - part1@part2
            diff = np.max(abs(beta_old-beta_new))
            beta_old = beta_new
            llh_vals.append(self.__loglikelihood(beta_new, gamma_cand, X_dat, Ti_list, Si_list))
            step_ind +=1

        return llh_vals, beta_new

    def __update_beta_LinReg(self, Si_list, X_dat, Ti_list, gamma_init):
        sigma_list = [Si_list[i]/Ti_list[i] for i in range(len(Si_list))]
        Z_arr = np.squeeze(np.array([gamma_init.T@sigma_list[i]@gamma_init for i in range(len(sigma_list))]))
        regmodel = LinearRegression().fit(X_dat[:,1:], np.log(Z_arr))
        beta_new = np.expand_dims(np.concatenate(([regmodel.intercept_], regmodel.coef_)),1)
        llh_vals = [self.__loglikelihood(beta_new, gamma_init, X_dat, Ti_list, Si_list)]

        return llh_vals, beta_new


    def __loglikelihood(self, beta, gamma, X_dat, Ti_list, Si_list):
        #Ignoring constants
        Xi_list = fca.make_Xi_list(X_dat)

        arr1 = np.array([(Xi_list[i].T@beta)*Ti_list[i] for i in range(X_dat.shape[0])])
        arr2 = np.array([gamma.T@Si_list[i]@gamma * np.exp(-Xi_list[i].T@beta) for i in range(X_dat.shape[0])])
        llh_value = (0.5 * np.sum(arr1) + 0.5 * np.sum(arr2))

        return llh_value

    def __random_initial_conds(self, p_model, n_init, seed=None):
        rng = np.random.default_rng(seed=seed)
        gamma_inits = rng.standard_normal([p_model,n_init])

        return gamma_inits

    def __deviation_from_diag(self, gamma_dir, Y_dat, weighted_io = 1, dfd_aritm = 0, logtrick_io = 1, FC_mode = False, Ti_list = []):
        """
        Computes the  "deviation from diagonality" (Flury and Gautschi, 1986) averaged across subjects for the projection specified by proj_mat. This function is called in the function DFD_values to determine the DFD_values sequentially for increasing number of gamma projections.
        
        Parameters:
        -----------
        gamma_dir: The direction up to which the dfd value is calculated. 
        Y_dat: List of length [number of subjects] containing time series data for each subject. Each element of the list should be array-like of shape (T[i], p), with T[i] the number of time points for subject i and p the number of regions/time series.
        weighted_io: Boolean or 0/1. If set to True/1, the average dfd value is a weighted average according to the number of time points for each subject. If all subjects have the same number of time points, setting this parameter to 1 can improve estimation by reducing the risk of overflow.
        dfd_aritm: Boolean or 0/1. If set to False/0, the average dfd value is computed as the geometric mean (weighted or unweighted according to the weighted_io parameter). If 1, the arithmetic mean is used. Default value is 0. 
        logtrick_io: Boolean of 0/1. When computing the harmonic mean, a log-transformation is temporarily applied to avoid overflow. Recommended, but can be disabled to test the difference. Default value 1.
        
        Returns: 
        --------
        dfd_proj: The average DfD value of the projected data matrices. 
        """

        gamma_here = self.gamma[:,:gamma_dir+1]

        if FC_mode == False:
            Si_list = fca.make_Si_list(Y_dat)
            Ti_list = np.array([Y_dat[i].shape[0] for i in range(len(Si_list))])
        else:
            Si_list = Y_dat
            Ti_list = np.array(Ti_list)

        mat_prods = [(gamma_here.T@Si_list[i]@gamma_here)/(Y_dat[i].shape[0]) for i in range(len(Si_list))]
        if not weighted_io:
            nu_vals = np.array([fca.dfd_func(mat_prods[i]) for i in range(len(Si_list))])
            if not dfd_aritm:
                if not logtrick_io:
                    dfd_proj = np.prod(nu_vals**(1/len(Y_dat)))
                elif logtrick_io:
                    dfd_proj = np.exp(np.mean(np.log(nu_vals)))
            else:
                dfd_proj = np.sum(nu_vals)/len(Y_dat)
        else:
            if not dfd_aritm:
                if not logtrick_io:
                    nu_vals = np.array([fca.dfd_func(mat_prods[i])**(Y_dat[i].shape[0]) for i in range(len(Si_list))])
                    sum_of_Ti = np.sum(Ti_list)
                    dfd_proj = (np.prod(nu_vals**(1/sum_of_Ti)))
                elif logtrick_io:
                    nu_vals = np.array([np.log(fca.dfd_func(mat_prods[i]))*(Y_dat[i].shape[0]) for i in range(len(Si_list))])
                    sum_of_Ti = np.sum(Ti_list)
                    dfd_proj = np.exp((np.sum(nu_vals)/sum_of_Ti))
            else:
                nu_vals = np.array([fca.dfd_func(mat_prods[i])*(Y_dat[i].shape[0]) for i in range(len(Si_list))])
                sum_of_Ti = np.sum(Ti_list)
                dfd_proj = (np.sum(nu_vals))/sum_of_Ti
                
        return dfd_proj
    
    def __make_Y_tilde_list(self, Y_dat, gamma_mat, beta_mat):
        """Creates list of Y_tilde matrices to be used when identifying multiple components. These contain the data after removing the components already identified.
        """

        num_gammas = gamma_mat.shape[1]
        p_model = Y_dat[0].shape[1]

        Yhat_kth_list = [Y_dat[i] - Y_dat[i]@gamma_mat@gamma_mat.T for i in range(len(Y_dat))]
        
        del Y_dat

        Ytilde_mats = []
        for i in range(len(Yhat_kth_list)):
            U_mat, D_mat, V_mat = np.linalg.svd(Yhat_kth_list[i], full_matrices=False)
            diag_el = D_mat[:(p_model-num_gammas)]
            diag_el = np.append(diag_el, np.sqrt(np.exp(beta_mat[0,:])))
            Dtilde_mat = np.diag(diag_el)
            Ytilde_mats.append(U_mat@Dtilde_mat@V_mat)

        return Ytilde_mats

    def __make_Si_list_tilde(self, Y_dat, gamma_mat, beta_mat):
        
        n_chunk = 500
        n_subj = len(Y_dat)
        n_loop = n_subj // n_chunk
        n_rest = n_subj % n_chunk

        Si_list_tilde = []
        for k in range(n_loop):
            Ytilde_mats_chunk = self.__make_Y_tilde_list(Y_dat[(k*n_chunk):((k+1)*n_chunk)], gamma_mat, beta_mat)
            Si_list_tilde_chunk = fca.make_Si_list(Ytilde_mats_chunk)
            Si_list_tilde.extend(Si_list_tilde_chunk)

        if n_rest > 0:
            Ytilde_mats_chunk = self.__make_Y_tilde_list(Y_dat[(n_loop*n_chunk):], gamma_mat, beta_mat)
            Si_list_tilde_chunk = fca.make_Si_list(Ytilde_mats_chunk)
            Si_list_tilde.extend(Si_list_tilde_chunk)

        return Si_list_tilde
    
    def __make_Si_list_tilde_fromFC(self, FC_list, gamma_mat, beta_mat, Ti_list, ddof):

        num_gammas = gamma_mat.shape[1]
        # p_model = FC_list[0].shape[0]

        Si_list = fca.make_Si_list_from_FC_list(FC_list, Ti_list, ddof)

        # Si_hat_list_alt = [(Si_list[i] - gamma_mat@gamma_mat.T@Si_list[i] - Si_list[i]@gamma_mat@gamma_mat.T + 
        #                 gamma_mat@gamma_mat.T@Si_list[i]@gamma_mat@gamma_mat.T) for i in range(len(FC_list))]
     
        gamma_prod = gamma_mat@gamma_mat.T
        # _, eigvecs_gammaprod = np.linalg.eigh(gamma_prod)
        # eigvecs_gammaprod_used = eigvecs_gammaprod[:,-num_gammas:]

        Si_list_tilde = [(Si_list[i] - gamma_prod@Si_list[i] - Si_list[i]@gamma_prod + 
                        gamma_prod@Si_list[i]@gamma_prod + gamma_mat@np.diag(np.exp(beta_mat[0,:]))@gamma_mat.T) for i in range(len(FC_list))]

        # Si_list_tilde_flip = [(Si_list[i] - gamma_prod@Si_list[i] - Si_list[i]@gamma_prod + 
        #                 gamma_prod@Si_list[i]@gamma_prod + eigvecs_gammaprod_used@np.diag(np.flip(np.exp(beta_mat[0,:])))@eigvecs_gammaprod_used.T) for i in range(len(FC_list))]
        
        # if num_gammas==3:
        #     Si_list_tilde_perm = [(Si_list[i] - gamma_prod@Si_list[i] - Si_list[i]@gamma_prod + 
        #                 gamma_prod@Si_list[i]@gamma_prod + eigvecs_gammaprod_used@np.diag(np.flip(np.exp(beta_mat[0,[0,2,1]])))@eigvecs_gammaprod_used.T) for i in range(len(FC_list))]

        # Si_list_tilde_alt = []
        # for i in range(len(Si_hat_list_alt)):
        #     eigvals_hat, eigvecs_hat = np.linalg.eigh(Si_hat_list_alt[i])
        #     eigvals_hat_sorted = np.flip(eigvals_hat)
        #     eigvecs_hat_sorted = np.flip(eigvecs_hat, axis=1)
        #     eigs_mod = eigvals_hat_sorted[:(p_model-num_gammas)]
        #     eigs_mod = np.append(eigs_mod, np.exp(beta_mat[0,:]))
        #     Dtilde_mat = np.diag(eigs_mod)
        #     Si_list_tilde_alt.append(eigvecs_hat_sorted@Dtilde_mat@eigvecs_hat_sorted.T)

        # print('HERE:')
        # print([np.all(np.abs(Si_list_tilde_alt[i]-Si_list_tilde[i])<1e-4) for i in range(len(Si_list_tilde_alt))])
        # print([np.all(np.abs(Si_list_tilde_alt[i]-Si_list_tilde_flip[i])<1e-4) for i in range(len(Si_list_tilde_alt))])
        # if num_gammas==3:
        #     print([np.all(np.abs(Si_list_tilde_alt[i]-Si_list_tilde_perm[i])<1e-4) for i in range(len(Si_list_tilde_alt))])

        return Si_list_tilde


    def __sample_s2_coefficients(self, X_dat):
        #Only called after decomposing with betaLinReg=True

        u_pred = self.predict(X_dat=X_dat)
        u_residuals = self.u_training - u_pred
        n_dir = u_residuals.shape[1]
        s2 = np.zeros(n_dir)
        for i in range(n_dir):
            s2[i] = np.dot(u_residuals[:,i],u_residuals[:,i])/(X_dat.shape[0]-X_dat.shape[1])

        return s2
    
    def __calc_Qinv(self, X_dat):
        Q_mat = X_dat.T@X_dat
        Q_inv = np.linalg.inv(Q_mat)
        return Q_inv