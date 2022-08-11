import numpy as np
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from itertools import product
import pandas as pd

# Load files from parent folders
import os
import sys
try: sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: print("Cannot load testing module")
from resample_and_test import ShiftTester

# Help functions
def e(n=1, mu=0): 
    return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def cb(*args): 
    return np.concatenate(args, axis=1)  # Col bind
p = norm.pdf

sigma_eps = 2.0
# Simulate data from a Gaussian SCM
def scm(n):
    X1 = 1 + np.sqrt(3)*e(n)
    X2 = X1 + sigma_eps*e(n)
    X3 = - X1 + X2 + e(n)
    return cb(X1, X2, X3)

# Weight function
def get_weight(target_mean=1, fit=False):
    def weight(X): 
        if not fit:
            return(p(X[:,1], loc=target_mean, scale=sigma_eps/2)/p(X[:, 1], loc=X[:,0], scale=sigma_eps))
        else:
            model = sm.OLS(X[:,2], sm.add_constant(X[:,0])).fit()
            old_mean = model.fittedvalues
            old_scale = np.sqrt(model.scale)
            return(p(X[:,1], loc=target_mean, scale=old_scale/2)/p(X[:, 1], loc=old_mean, scale=old_scale))
    return weight


def get_outcome(fit=False):
    def mu(X):
        if not fit:
            return X[:,1] - X[:,0]
        else:
            return sm.OLS(X[:,2], sm.add_constant(X[:,[0,1]])).fit().fittedvalues
    return mu

# Test function
def get_T(fit=False):
    def T(X): 
        return 1*(sm.stats.ztest(X[:, 2], value=0)[1] < 0.05)
    return T
# IPW test function
def get_T_ipw(target_mean, fit=False):
    def T_ipw(X):
        n_ = X.shape[0]
        z = get_weight(target_mean, fit=fit)(X)*X[:,2]
        ipw_estimate = z.mean()
        sd = np.sqrt(z.var())
        return 1-(np.abs(ipw_estimate) < 1.96*sd/np.sqrt(n_))
    return T_ipw
# Clipped IPW test function
def get_T_ipw_clipped(target_mean, fit=False):
    def T_ipw_clipped(X):
        n_ = X.shape[0]
        z = get_weight(target_mean,fit=fit)(X)

        # Clip weights
        z = np.minimum(z, z[np.argsort(z)[-10:]].min())
        z = z*X[:,2]
        ipw_estimate = z.mean()
        sd = np.sqrt(z.var())

        return 1-(np.abs(ipw_estimate) < 1.96*sd/np.sqrt(n_))
    return T_ipw_clipped

# Doubly robust test function
def get_T_doubly_robust(target_mean, fit=False):
    def T_doubly_robust(X):
        n_ = X.shape[0]
        z = get_weight(target_mean, fit)(X)
        
        # Outcome model
        mu = get_outcome(fit)
        dr_summands = (mu(X) + z*(X[:,2] - mu(X)))
        dr_estimate = dr_summands.mean()
        dr_sd = np.sqrt(dr_summands.var())

        return 1-(np.abs(dr_estimate) < 1.96*dr_sd/np.sqrt(n_))
    return T_doubly_robust

# Cutoff for tuning of m
cutoff = np.quantile(np.random.uniform(size=(1000, 15)).min(axis=1), 0.05)

# Loop parameters
target_mean_range = np.linspace(1, 6, num=11)
test_range = {"Resampling": get_T, "IPW": get_T_ipw, "IPWClipped": get_T_ipw_clipped, "DR": get_T_doubly_robust}
fit_range = [True, False]
n = 100

combinations = list(product(target_mean_range, test_range.keys(), fit_range))


def conduct_experiment(i=None):
    np.random.seed(i)
    out = []
    for target_mean, test, fit in combinations:
        X = scm(n)

        if test =="Resampling":
            try:
                psi = ShiftTester(get_weight(target_mean, fit=fit), test_range[test](fit), rate=None, replacement="NO-REPL-reject", reject_retries=100)
                m = psi.tune_m(X, j_x=[0], j_y=[1], gaussian=True,
                               cond=[0], const=target_mean,
                               m_factor=1.3, p_cutoff=cutoff, replacement=False,
                               m_init=int(np.sqrt(n)), repeats=15)
                out.append(psi.test(X, m=m))
            except:
                # Catch errors from test statistic
                print(f"Error occurred {target_mean}, {n}, {test}")
                out.append(np.nan)
        else:
            try:
                out.append(test_range[test](target_mean=target_mean, fit=fit)(X))
            except:
                out.append(np.nan)
    return out

## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 1000

    # Multiprocess
    pool = Pool(cpu_count()-2)
    res = np.array(
        list(tqdm(pool.imap_unordered(conduct_experiment, range(repeats)), total=repeats)))
    pool.close()

    # Count non-nas, to be used for binomial confidence intervals
    counts = (~np.isnan(res)).sum(axis=0)
    res = np.nansum(res, axis=0)

    # Pack as data frame
    df = pd.DataFrame(
        [(x/c, *v, *proportion_confint(x, c, method="binom_test"), c) for x, v, c in zip(res, combinations, counts)],
        columns=["RejectRate", "TargetMean", "Test", "Fit", "Lower", "Upper", "Count"])

    # Export to R for ggplotting
    df['RejectRate'] = df["RejectRate"].replace(np.NaN, "NA")
    print(df)
    df.to_csv("experiments/section-5-5/compare-to-ipw.csv")
