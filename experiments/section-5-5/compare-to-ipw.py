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
def e(n=1, mu=0): return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def cb(*args): return np.concatenate(args, axis=1)  # Col bind
p = norm.pdf
np.random.seed(1)

sigma_eps = 2.0
# Simulate data from a Gaussian SCM
def scm(n):
    X1 = 1 + np.sqrt(3)*e(n)
    X2 = X1 + sigma_eps*e(n)
    X3 = - X1 + X2 + e(n)
    return cb(X1, X2, X3)

# Weight function
def get_weight(target_mean=1):
    def weight(X): return(p(X[:,1], loc=target_mean, scale=sigma_eps/2)/p(X[:, 1], loc=X[:,0], scale=sigma_eps))
    return weight

# Test function
def T(X): return 1*(sm.stats.ztest(X[:, 2], value=0)[1] < 0.05)

# IPW test function
def T_ipw(X, target_mean):
    n_ = X.shape[0]
    z = get_weight(target_mean)(X)*X[:,2]
    ipw_estimate = z.mean()
    sd = np.sqrt(z.var())
    return 1-(np.abs(ipw_estimate) < 1.96*sd/np.sqrt(n_))

# Clipped IPW test function
def T_ipw_clipped(X, target_mean):
    n_ = X.shape[0]
    z = get_weight(target_mean)(X)

    # Clip weights
    z = np.minimum(z, z[np.argsort(z)[-10:]].min())
    z = z*X[:,2]
    ipw_estimate = z.mean()
    sd = np.sqrt(z.var())

    return 1-(np.abs(ipw_estimate) < 1.96*sd/np.sqrt(n_))

# Cutoff for tuning of m
cutoff = np.quantile(np.random.uniform(size=(1000, 15)).min(axis=1), 0.05)

# Loop parameters
target_mean_range = np.linspace(1, 6, num=11)
test_range = {"Resampling": T, "IPW": T_ipw, "IPWClipped": T_ipw_clipped}
n = 100

combinations = list(product(target_mean_range, test_range.keys()))


def conduct_experiment(i=None):
    out = []
    for target_mean, test in combinations:
        X = scm(n)

        if test =="Resampling":
            try:
                psi = ShiftTester(get_weight(target_mean), test_range[test], rate=None, replacement="NO-REPL-reject", reject_retries=100)
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
                out.append(test_range[test](X, target_mean))
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
        columns=["RejectRate", "TargetMean", "Test", "Lower", "Upper", "Count"])

    # Export to R for ggplotting
    df['RejectRate'] = df["RejectRate"].replace(np.NaN, "NA")
    df.to_csv("experiments/section-5-5/compare-to-ipw.csv")
