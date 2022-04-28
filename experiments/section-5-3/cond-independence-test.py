import numpy as np
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.gam.api import GLMGam, BSplines
from tqdm import tqdm
from itertools import product
import pandas as pd
import torch
from statsmodels.stats.proportion import proportion_confint
import rpy2.robjects.packages as packages

# Import rpy for testing with R functions
import rpy2.robjects as robj
CondIndTests = packages.importr("CondIndTests")
GeneralisedCovarianceMeasure = packages.importr("GeneralisedCovarianceMeasure")
dHSIC = packages.importr("dHSIC")

# Load files from parent folders
import os
import sys
try: sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: print("Cannot load testing module")
from resample_and_test import ShiftTester

# Help functions
alpha = 3
def e(n=1, mu=0): return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def coin(n, p=0.5): return 1*(np.random.uniform(size=(n,1)) < 0.5)
def cb(*args): return np.concatenate(args, axis=1)  # Col bind
def to_r(x): return robj.FloatVector(x)

np.random.seed(1)

# Simulate data from a Gaussian SCM
def scm(n, causal_effect=1, causal_exponent=2):
    X1 = e(n, mu=4*coin(n)-2) # Bimodal Gaussian distribution
    X2 = -X1**2 + 2*e(n)
    X3 = np.sin(X2) + causal_effect*X1**causal_exponent + 2*e(n)
    return cb(X1, X2, X3)

p = norm.pdf

# Weight
def weight(X):
    return p(X[:,1], loc = X[:,1].mean(), scale=np.sqrt(X[:,1].var()))/p(X[:,1], loc=-(X[:,0])**2, scale=2)

# Fitted weight
def weight_fit(X):
    mod1 = GLMGam(X[:,1], smoother=BSplines(X[:,0], df = 10, degree=3)).fit()
    return p(X[:,1], loc = X[:,1].mean(), scale=np.sqrt(X[:,1].var()))/p(X[:,1], loc=mod1.fittedvalues, scale=np.sqrt(mod1.scale))


# Test function: LinReg
def T(X):
    return (sm.OLS(X[:, 2], sm.tools.add_constant(X[:, 0])).fit().pvalues[1] < 0.05)*1

# HSIC test statistic
def THSIC(X):
    pval = dHSIC.dhsic_test(X=to_r(X[:,0]), Y=to_r(X[:,2])).rx2("p.value")[0]
    return 1*(pval < 0.05)

# Kernel Conditional Independence
def KCI(X):
    pval = (CondIndTests.KCI(to_r(X[:,0]), to_r(X[:,2]), to_r(X[:,1])).rx2('pvalue'))[0]
    return 1*(pval < 0.05)

# GeneralisedCovarianceMeasure
def GCM(X):
    pval = (GeneralisedCovarianceMeasure.gcm_test(X=to_r(X[:,0]), Z=to_r(X[:,1]), Y=to_r(X[:,2]), regr_method="gam").rx2('p.value'))[0]
    return 1*(pval < 0.05)

# Loop parameters
causal_effects = np.linspace(0, 1.5, num=7)
causal_exponents = [1, 2]
n = 150
tests = {"HSIC": THSIC, "LinReg": T, "HSICfit": THSIC, "GCM": GCM, "KCI": KCI}

# Compute for goodness-of-fit quantile
cutoff = np.quantile(np.random.uniform(size=(1000, 15)).min(axis=1), 0.005)

# Conditional distribution for goodness-of-fit
def get_cond(loc, scale):
    def cond(X, Y):
        slope = torch.tensor([0])
        linear = loc
        return -(Y.view(-1)-linear)**2/(2.0*scale**2)
    return cond

combinations = list(product(causal_effects, causal_exponents, tests.keys()))

## Wrap as function to apply multiprocessing
def conduct_experiment(i=None):
    out = []
    for c_e, c_exp, test in combinations:
        X = scm(n, causal_effect=c_e, causal_exponent=c_exp)
        if not test in ["GCM", "KCI"]:
            try:
                if test == "HSICfit":
                    psi = ShiftTester(weight_fit, tests[test], rate=None, replacement="REPL-reject", reject_retries=500)
                else:
                    psi = ShiftTester(weight, tests[test], rate=None, replacement="REPL-reject", reject_retries=500)
                m = psi.tune_m(X, j_x=[0], j_y=[1], gaussian=False,
                               cond=get_cond(loc=X[:,1].mean(), scale=np.sqrt(X[:,1].var())),
                               m_factor=1.3, p_cutoff=cutoff, replacement=False,
                               m_init=int(np.sqrt(n)), repeats=15)
                out.append(psi.test(X, m=m))
            except:
                # Catch errors from test statistic
                print(f"Error occurred {test}, {n}")
                out.append(np.nan)
        else:
            try:
                out.append(tests[test](X))
            except:
                print(f"Error occurred {test}, {n}")
                out.append(np.nan)
    return(out)

## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 500

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
        columns=["alpha", "CausalEffect", "EffectOrder", "Test", "Lower", "Upper", "Count"])

    # Export to R for ggplotting
    df['alpha'] = df["alpha"].replace(np.NaN, "NA")
    df.to_csv("experiments/section-5-3/cond-independence-test.csv")
