import numpy as np
from matplotlib import pyplot as plt

"""
Basic plotting functions for Functional Connectivity Integrative Normative Modelling (FUNCOIN)
@Author and maintainer of Python package: Janus Rønn Lind Kobbersmed, janus@cfin.au.dk or januslind@gmail.com
@Based on the Covariate-Assisted Principal regression method: Zhao, Y. et al. (2021). 'Covariate Assisted Principal regression for covariance matrix outcomes', Biostatistics, 22(3), pp. 629-45.  
"""

def visualize_gamma_coefs(Fcn, n_comps = 1, threshold = False):
    """For each component and illustration of the gamma coefficients is plotted
        
    Parameters:
    -----------
    Fcn: A fitted instance of the Funcoin class. Can also be the gamma matrix directly. 
        If inputting gamma, it needs to be a numpy array of shape [no. of regions] by [no. of components]
    n_comps: Number of components to visualize. Default value is 1.
    threshold (optional): Adds vertical dotted lines to indicate a threshold on the coefficients.
    
    Returns:
    --------
    Plots the gamma coefficients for each region.
    """
    
    if hasattr(Fcn, 'gamma'):
        gamma_mat = Fcn.gamma
    elif type(Fcn) ==  np.ndarray:
        gamma_mat = Fcn

    p_model = gamma_mat.shape[0]

    for i in range(n_comps):
        fig, ax = plt.subplots(1)
        ax.plot([0,0], [1, p_model], 'k', linewidth=3)

        if threshold:
            ax.plot([-threshold,-threshold], [1, p_model], 'k--')
            ax.plot([threshold,threshold], [1, p_model], 'k--')

        for k in range(p_model):
            if gamma_mat[k,i]>0:
                col = 'r'
            else:
                col = 'b'

            ax.plot(np.array([0, gamma_mat[k,i]]), np.array([k+1,k+1]), col, linewidth= 2)

        ax.set_yticks(np.arange(p_model)+1)
        ax.set_xlabel('gamma loading')
        ax.set_ylabel('Region number')
        plt.tight_layout()
        plt.show()
