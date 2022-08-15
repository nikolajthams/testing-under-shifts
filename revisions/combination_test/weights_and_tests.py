import statsmodels.api as sm
import numpy as np
from statsmodels.gam.api import GLMGam, BSplines
from scipy.stats import norm
p = norm.pdf


# Import rpy for testing with R functions
import rpy2.robjects.packages as packages
import rpy2.robjects as robj
dHSIC = packages.importr("dHSIC")
def to_r(x): return robj.FloatVector(x)



# Weights
def weight_fitted(X, linear=False, reweight_target=1, reweight_covariate=0, target_var_reduction_factor=1):
    # Regress to learn old mean and scale
    if linear:
        model = sm.OLS(X[:,reweight_target], sm.tools.add_constant(X[:,reweight_covariate])).fit()
    else: 
        model = GLMGam(X[:,reweight_target], smoother=BSplines(X[:,reweight_covariate], df = 10, degree=3, include_intercept=True)).fit()
    old_mean = model.fittedvalues
    old_scale = np.sqrt(model.scale)

    # Set target mean and scale as marginal distribution
    new_mean = X[:,reweight_target].mean()
    new_scale = np.sqrt(X[:,reweight_target].var())*target_var_reduction_factor
    return p(X[:,reweight_target], loc=new_mean, scale=new_scale)/p(X[:,reweight_target], loc=old_mean, scale=old_scale)

def get_weight_func(linear=False, reweight_target=1, reweight_covariate=[0], target_var_reduction_factor=1):
    def weight_func(X):
        return weight_fitted(X, linear=linear, reweight_target=reweight_target, reweight_covariate=reweight_covariate, target_var_reduction_factor=target_var_reduction_factor)
    return weight_func

# Function for getting raw p-values
def get_p_val_func(response_pos=2, covariate_pos=0):
    def p_val_func(X):
        return sm.OLS(X[:, response_pos], sm.tools.add_constant(X[:, covariate_pos])).fit().pvalues[1]
    return p_val_func

# Function for getting test statistic.
def get_T(causal_effect=0, alpha=0.05, test_conf_int=False, response_pos=2, covariate_pos=0):
    # Test hypothesis of 0 correlation
    def T(X):
        return 1.0*(sm.OLS(X[:, response_pos], sm.tools.add_constant(X[:, covariate_pos])).fit().pvalues[1]<alpha)
    # Test hypothesis that regression slop is the specified causal effect
    def T_CI(X):
        lower, upper = sm.OLS(X[:, response_pos], sm.tools.add_constant(X[:, covariate_pos])).fit().conf_int(alpha=alpha)[1]
        return 1.0*((lower > causal_effect) | (causal_effect > upper))
    return T_CI if test_conf_int else T

# HSIC test statistic
def get_HSIC(args, return_p_val=False):
    def THSIC(X):
        pval = dHSIC.dhsic_test(X=to_r(X[:,args.covariate_pos]), Y=to_r(X[:,args.response_pos])).rx2("p.value")[0]
        if return_p_val:
            return pval
        else:
            return 1.0*(pval < args.alpha)
    return THSIC

def permutation_test(X, Y, n_permutations=1000, seed=None, alpha=None):
    if seed is not None:
        np.random.seed(seed)

    def test_statistic(x, y):
        return np.abs(np.corrcoef(x, y)[0,1])
    
    # Get p-values
    test_statistics = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Get p-value
        Y_shuffled = Y.copy()
        np.random.shuffle(Y_shuffled)
        test_statistics[i] = test_statistic(X, Y_shuffled)
    p_val =  ((test_statistic(X, Y) < test_statistics).sum() + 1)/(1 + test_statistics.size)
    if alpha is not None:
        return 1.0*(p_val < alpha)
    return p_val

def get_permutation_p_val(n_permutations=1000, response_pos=2, covariate_pos=0):
    def p_val_func(data):
        return permutation_test(data[:,covariate_pos], data[:,response_pos], n_permutations=n_permutations)
    return p_val_func

def get_permutation_T(n_permutations=1000, response_pos=2, covariate_pos=0, alpha=0.05):
    def T(data):
        return permutation_test(data[:,covariate_pos], data[:,response_pos], n_permutations=n_permutations, alpha=alpha)
    return T