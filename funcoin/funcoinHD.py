from .funcoin import Funcoin
from sklearn.covariance import ledoit_wolf

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

    def __str__(self):
        firststr = 'Instance of the high-dimensional case of Functional Connectivity Integrative Normative Modelling (FUNCOIN) class. '

        laststr = self.__create_fitstring()

        return firststr + laststr


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

        try:
            add_to_fit = kwargs['add_to_fit']
        except:
            add_to_fit = False

        super().__initialise_decomposition(self, max_comps=max_comps, gamma_init = gamma_init, rand_init = rand_init, n_init = n_init, max_iter = max_iter, tol=tol, trace_sol = trace_sol, seed_initial = seed_initial, betaLinReg = betaLinReg, overwrite_fit = overwrite_fit, add_to_fit = add_to_fit)



        ##Implement fitting routine here