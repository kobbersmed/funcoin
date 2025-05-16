from .funcoin import Funcoin
from . import funcoin_auxiliary as fca
import numpy as np
import warnings

class FuncoinHD(Funcoin):
    """
    Class for the high-dimensional version of Functional Connectivity Integrative Normative Modelling (FUNCOIN).

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
        super().__init__(gamma=gamma, beta=beta)
        self.mu = float('nan')
        self.rho = float('nan')

    def __str__(self):
        firststr = 'Instance of the high-dimensional case of Functional Connectivity Integrative Normative Modelling (FUNCOIN) class. '

        laststr = super().__create_fitstring()

        return firststr + laststr


    def decompose(self, Y_dat, X_dat, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, tol_shrinkage = 1e-4, trace_sol = 0, seed_initial = None, betaLinReg = True, overwrite_fit = False, **kwargs):
        """Performs FUNCOIN decomposition for high dimensional covariance matrices given a list of time series data, Y_dat, and covariate matrix, X_dat. 
        
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
        tol: Float >0. Maximal tolerance when optimizing for gamma (and beta). If an iteration yields an absolute change smaller than this value in all elements of gamma and beta and the shrinkage parameter tolerance is also satisfied, the optimisation stops. Default 1e-4.
        tol_shrinkage: Float >0. Maximal tolerance when optimizing for the shrinkage parameters mu and rho. If an iteration yields an absolute change in mu and rho smaller than this value and the beta and gamma tolerance is also satisfied, the optimisation stops. Default 1e-4.
        trace_sol: Boolean. Whether or not to keep all intermediate steps in the optimization of gamma and beta. The steps are stored in lists in instance variables self.gamma_steps_all and self.beta_steps_all.
        seed_initial: Integer or None. If integer, this seeds the random initial conditions. Default value False.
        betaLinReg: Boolean. If true, the algorithm concludes with performing ordinary linear regression on the transformed values using the gamma transformation found to improve accuracy of beta estimation. Default False.
        overwrite_fit: Boolean. If False: Returns an exception if the class object has already been fitted to data. If True: Fits using the provided data and overwrites any existing values of gamma, beat, dfd_values_training, and u_training.

        Returns:
        --------

        self.beta: Array-like of shape (q,n_dir). Coefficients of the log-linear model identified during decomposition.
        self.gamma: Array-like of shape (p,n_dir). Matrix with each column being an identified gamma projection.
        self.mu: Vector of length n_dir. Contains the fitted mu values for each direction from the shrinkage procedure. The "shrinkage target" is mu*I, where I is the p-by-p identity matrix.
        self.rho: Vector of length n_dir. Contains the fitted rho values (shrinkage weights) for each direction from the shrinkage procedure. The shrunk covariance matrix for subject i is rho*mu*I + (1-rho)*S_i, 
                    where S_i is the sample covariance matrix of subject i, mu is the shrinkage weight, and I is the p-by-p identity matrix. 
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

        try:
            add_to_fit = kwargs['add_to_fit']
        except:
            add_to_fit = False

        super().__store_decomposition_options(self, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed_initial = seed_initial, betaLinReg = betaLinReg, overwrite_fit = overwrite_fit, add_to_fit = add_to_fit, tol_shrinkage=tol_shrinkage)

        beta_mat, gamma_mat, mu_vec, rho_vec = self.__decompositionHD(Y_dat, X_dat, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, tol_shrinkage = tol_shrinkage, trace_sol = trace_sol, seed = seed_initial, betaLinReg = betaLinReg, overwrite_fit = overwrite_fit, add_to_fit = add_to_fit, FC_mode = False, Ti_list=[], ddof = 0)
        
        super().__store_fitresult(Y_dat, X_dat, gamma_mat, beta_mat, betaLinReg, FC_mode = False, Ti_equal = [], mu=mu_vec, rho=rho_vec)

    def decompose_FC(self, FC_list, X_dat, Ti_list, ddof = 0, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, tol_shrinkage=1e-4, trace_sol = 0, seed_initial = None, betaLinReg = True, overwrite_fit = False, **kwargs):
        """Performs FUNCOIN decomposition for high dimensional FC matrices given a list of FC matrices, FC_list, a covariate matrix, X_dat, and a list of the number of time points in the original time series data. 
        
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
        self.mu: Vector of length n_dir. Contains the fitted mu values for each direction from the shrinkage procedure. The "shrinkage target" is mu*I, where I is the p-by-p identity matrix.
        self.rho: Vector of length n_dir. Contains the fitted rho values (shrinkage weights) for each direction from the shrinkage procedure. The shrunk covariance matrix for subject i is rho*mu*I + (1-rho)*S_i, 
                    where S_i is the sample covariance matrix of subject i, mu is the shrinkage weight, and I is the p-by-p identity matrix. 
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
        elif type(Ti_list) == list:
            if len(Ti_list)!=len(FC_list):
                raise Exception('Length of list of FC matrices and list of number of time points do not match')



        try:
            add_to_fit = kwargs['add_to_fit']
        except:
            add_to_fit = False

        super().__store_decomposition_options(max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed_initial = seed_initial, betaLinReg = betaLinReg, overwrite_fit = overwrite_fit, add_to_fit = add_to_fit, tol_shrinkage = tol_shrinkage)

        beta_mat, gamma_mat, mu_vec, rho_vec = self.__decompositionHD(FC_list, X_dat, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, tol_shrinkage = tol_shrinkage, trace_sol = trace_sol, seed = seed_initial, betaLinReg = betaLinReg, overwrite_fit = overwrite_fit, add_to_fit = add_to_fit, FC_mode = False, Ti_list=[], ddof = 0)
       
        super().__store_fitresult(FC_list, X_dat, gamma_mat, beta_mat, betaLinReg, FC_mode = True, Ti_list=Ti_list, mu=mu_vec, rho = rho_vec)


    def decompose_ts(self, Y_dat, X_dat, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, tol_shrinkage = 1e-4, trace_sol = 0, seed_initial = None, betaLinReg = True, overwrite_fit = False, **kwargs):
        """Performs FUNCOIN decomposition for high dimensional covariance matrices given a list of time series data, Y_dat, and covariate matrix, X_dat. 
        
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
        tol: Float >0. Maximal tolerance when optimizing for gamma (and beta). If an iteration yields an absolute change smaller than this value in all elements of gamma and beta and the shrinkage parameter tolerance is also satisfied, the optimisation stops. Default 1e-4.
        tol_shrinkage: Float >0. Maximal tolerance when optimizing for the shrinkage parameters mu and rho. If an iteration yields an absolute change in mu and rho smaller than this value and the beta and gamma tolerance is also satisfied, the optimisation stops. Default 1e-4.
        trace_sol: Boolean. Whether or not to keep all intermediate steps in the optimization of gamma and beta. The steps are stored in lists in instance variables self.gamma_steps_all and self.beta_steps_all.
        seed_initial: Integer or None. If integer, this seeds the random initial conditions. Default value False.
        betaLinReg: Boolean. If true, the algorithm concludes with performing ordinary linear regression on the transformed values using the gamma transformation found to improve accuracy of beta estimation. Default False.
        overwrite_fit: Boolean. If False: Returns an exception if the class object has already been fitted to data. If True: Fits using the provided data and overwrites any existing values of gamma, beat, dfd_values_training, and u_training.

        Returns:
        --------

        self.beta: Array-like of shape (q,n_dir). Coefficients of the log-linear model identified during decomposition.
        self.gamma: Array-like of shape (p,n_dir). Matrix with each column being an identified gamma projection.
        self.mu: Vector of length n_dir. Contains the fitted mu values for each direction from the shrinkage procedure. The "shrinkage target" is mu*I, where I is the p-by-p identity matrix.
        self.rho: Vector of length n_dir. Contains the fitted rho values (shrinkage weights) for each direction from the shrinkage procedure. The shrunk covariance matrix for subject i is rho*mu*I + (1-rho)*S_i, 
                    where S_i is the sample covariance matrix of subject i, mu is the shrinkage weight, and I is the p-by-p identity matrix. 
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

        self.decompose(self, Y_dat, X_dat, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, tol_shrinkage = tol_shrinkage, trace_sol = trace_sol, seed_initial = seed_initial, betaLinReg = betaLinReg, overwrite_fit = overwrite_fit, **kwargs)

    def __decompositionHD(self, Y_dat, X_dat, max_comps=2, gamma_init = False, rand_init = True, n_init = 20, max_iter = 1000, tol=1e-4, tol_shrinkage = 1e-4, trace_sol = 0, seed = None, betaLinReg = True, overwrite_fit = False, add_to_fit = False, FC_mode = False, Ti_list=[], ddof = 0):
        
        n_subj = len(Y_dat)

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

        if FC_mode == False:
            Y_dat = [Y_dat[i]-np.mean(Y_dat[i],0) for i in range(len(Y_dat))]
            Ti_list = [Y_dat[i].shape[0] for i in range(len(Y_dat))]
            Si_list = fca.make_Si_list(Y_dat)
        else:
            Si_list = fca.make_Si_list_from_FC_list(Y_dat, Ti_list, ddof)
        
        p_model = Si_list[0].shape[0]

        for i_dir in range(n_dir_init,max_comps):
            
            gamma_init_used = super().__initialise_gamma(gamma_init, rand_init, p_model, n_init, seed)

            if i_dir == 0:
                try:
                    _, best_beta, best_gamma, best_mu, best_rho = self.__first_directionHD(Si_list, X_dat, Ti_list, gamma_init_used, max_iter = 1000, tol = 1e-4, tol_shrinkage = 1e-4, trace_sol = 0, betaLinReg = True)
                except:
                    raise Exception('Exception occured. Did not find any principal directions using FUNCOIN algorithm.')
                else:
                    beta_mat_new = best_beta
                    gamma_mat_new = best_gamma
                    mu_vec_new = best_mu
                    rho_vec_new = best_rho
            else:
                try:
                    beta_mat_new, gamma_mat_new, _, _, _ = self.__kth_directionHD(Y_dat, X_dat, beta_mat, gamma_mat, mu_vec, rho_vec, gamma_init_used, max_iter=max_iter, tol = tol, trace_sol=trace_sol, betaLinReg=betaLinReg, FC_mode = FC_mode, Ti_list=Ti_list, ddof = ddof)
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
            mu_vec = mu_vec_new
            rho_vec = rho_vec_new

        return beta_mat, gamma_mat, mu_vec, rho_vec

    def __first_directionHD(self, Si_list, X_dat, Ti_list, gamma_init_used, max_iter = 1000, tol = 1e-4, tol_shrinkage = 1e-4, trace_sol = 0, betaLinReg = True):

        gammas_allinits = []
        betas_allinits = []
        rho_allinits = []
        mu_allinits = []
        llh_allinits = []

        n_init = gamma_init_used.shape[1]

        for i_init in range(n_init):
            gamma_init_here = np.expand_dims(gamma_init_used[:,i_init],1)
            gamma_old = gamma_init_here/np.linalg.norm(gamma_init_here)
            _, beta_old = super().__update_beta_LinReg(Si_list, X_dat, Ti_list, gamma_old)

            step_ind = 0
            shrink_diff = 100

            while (step_ind<max_iter) and (shrink_diff > tol_shrinkage):
                
                mu_new, rho_new = self.__calc_shrinkage_parameters(X_dat, gamma_old, beta_old, Si_list, Ti_list)

                Si_star_list = self.__create_Si_star_list(mu_new, rho_new, Si_list)
                
                best_llh, best_beta, best_gamma, _, _, _, _, _, _, _, _ = super().__first_direction(Si_star_list, X_dat, Ti_list, gamma_init = gamma_old, max_iter = max_iter, tol = tol, trace_sol = trace_sol, betaLinReg = False)
                
                rho_diff = np.abs(rho_new-rho_old)
                mu_diff = np.abs(mu_new-mu_old)
                shrink_diff = np.maximum(rho_diff, mu_diff)
                
                rho_old = rho_new
                mu_old = mu_new
                gamma_old = best_gamma
                beta_old = best_beta
            
            if betaLinReg:
                _, beta_new = super().__update_beta_LinReg(Si_star_list, X_dat, Ti_list, gamma_old)
            else:
                _, beta_new = super().__optimize_only_beta(Si_star_list, X_dat, Ti_list, beta_old, gamma_old)
        

            gammas_allinits.append(gamma_old)
            betas_allinits.append(beta_new)
            rho_allinits.append(rho_old)
            mu_allinits.append(mu_old)
            llh_allinits.append(best_llh)

        best_llh_ind = np.argmin(llh_allinits)
        best_llh = llh_allinits[best_llh_ind]
        best_gamma = gammas_allinits[best_llh_ind]
        best_beta = betas_allinits[best_llh_ind]
        best_rho = mu_allinits[best_llh_ind]
        best_mu = rho_allinits[best_llh_ind]

        return best_llh, best_beta, best_gamma, best_mu, best_rho

    def __kth_directionHD(self, Y_dat, X_dat, beta_mat, gamma_mat, mu_vec, rho_vec,  gamma_init=False, max_iter=1000, tol=1e-4, trace_sol=0, betaLinReg=False, FC_mode=False, Ti_list=..., ddof=0):
        if FC_mode == False:
            Ti_list = [Y_dat[i].shape[0] for i in range(len(Y_dat))]
            Si_list_tilde = super().__make_Si_list_tilde(Y_dat, gamma_mat, beta_mat)
        else:
            Si_list_tilde = super().__make_Si_list_tilde_fromFC(Y_dat, gamma_mat, beta_mat, Ti_list, ddof)

        best_llh, best_beta, best_gamma, best_mu, best_rho = self.__first_directionHD(Si_list_tilde, X_dat, Ti_list, gamma_init, max_iter = 1000, tol = 1e-4, tol_shrinkage = 1e-4, trace_sol = 0, betaLinReg = True)
        
        gamma_mat_new = np.append(gamma_mat, best_gamma, 1)
        beta_mat_new = np.append(beta_mat, best_beta, 1)
        mu_vec_new = np.append(mu_vec, best_mu)
        rho_vec_new = np.append(rho_vec, best_rho)

        return beta_mat_new, gamma_mat_new, best_llh, mu_vec_new, rho_vec_new
  
    def __calc_shrinkage_parameters(self, X_dat, gamma_vec, beta_vec, Si_list, Ti_list):
        n_subj = X_dat.shape[0]
        Xi_list = fca.make_Xi_list(X_dat)
        exp_xi_beta_array = np.array([Xi_list[i].T@np.exp(beta_vec) for i in range(n_subj)]) 
        exp_x_beta_sum = np.sum(exp_xi_beta_array)

        transf_Si_array = np.array([gamma_vec.T@Si_list[i]@gamma_vec for i in range(n_subj)])

        mu = (1/(n_subj * (gamma_vec.T@gamma_vec)))*exp_x_beta_sum
        # phi_i_sq = np.array([mu*(gamma_vec.T@gamma_vec)-exp_xi_beta_array[i] for i in range(len(Xi_list))])**2
        # phi_sq = (1/n_subj) * np.sum(phi_i_sq)
        
        deltahat_i_sq = np.array([transf_Si_array[i]-mu*(gamma_vec.T@gamma_vec) for i in range(n_subj)])**2
        deltahat_sq = (1/n_subj)*np.sum(deltahat_i_sq)

        psihat_i_sq = np.array([(1/Ti_list[i]) * (transf_Si_array[i] - exp_xi_beta_array[i]) for i in range(n_subj)])**2
        psihat_sq = (1/n_subj)*np.sum(np.array([np.minimum(psihat_i_sq[i], deltahat_i_sq[i]) for i in range(n_subj)]))

        rho = psihat_sq/deltahat_sq
        
        
        return mu, rho

    def __create_Si_star_list(self, mu, rho, Si_list):
        p_model = Si_list[0].shape[0]
        Si_star_list = [rho*mu*np.identity(p_model)+(1-rho)*Si_list[i] for i in range(len(Si_list))]

        return Si_star_list
    